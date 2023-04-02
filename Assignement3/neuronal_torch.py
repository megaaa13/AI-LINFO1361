import numpy as np
import logging
import string
import random
import copy

# Pontu
from pontu_state import PontuState
from agent import Agent

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Agents
from random_agent import MyAgent as RandomAgent
from basic_agent import MyAgent as BasicAgent

# Multi-processing
import concurrent.futures
import threading

# Logging configuration
logging.basicConfig(filename="games.log", level=logging.INFO, filemode="w")


####################
# Neuronal Helpers #
####################

class NeuralHelpers:
    '''
    Helper functions for neural networks
    '''

    @staticmethod
    def state_to_features(state: PontuState) -> np.ndarray:
        '''
        Returns a list of features for the current state.
        The features are listed in the following order:
        - List of pawns coordinates
            - Pawns 1 of player 0
            - Pawns 1 of player 1
            - Pawns 2 of player 0
            - Pawns 2 of player 1
            - ...
        - List of horizontal bridges (boolean) (line by line, left to right, top to bottom)
        - List of vertical bridges (boolean) (line by line, left to right, top to bottom)
        '''
        num_pawns = (state.size - 2) * 2 * 2
        num_h_bridges = state.size * (state.size - 1)
        num_v_bridges = (state.size - 1) * state.size

        state_array = np.zeros(num_pawns + num_h_bridges + num_v_bridges, dtype=np.int8)

        # Pawns information 
        for i in range(state.size - 2):
            idx = i * 4
            state_array[idx]   = state.cur_pos[0][i][0]  # Pawn i of player 0, x coordinate
            state_array[idx+1] = state.cur_pos[0][i][1]  # Pawn i of player 0, y coordinate
            state_array[idx+2] = state.cur_pos[1][i][0]  # Pawn i of player 1, x coordinate
            state_array[idx+3] = state.cur_pos[1][i][1]  # Pawn i of player 1, y coordinate
        
        # Horizontal bridges information
        h_start_idx = num_pawns
        h_end_idx = h_start_idx + num_h_bridges
        state_array[h_start_idx:h_end_idx] = [1 if state.h_bridges[i][j] else 0 for i in range(state.size) for j in range(state.size - 1)]
        
        # Vertical bridges information
        v_start_idx = h_end_idx
        v_end_idx = v_start_idx + num_v_bridges
        state_array[v_start_idx:v_end_idx] = [1 if state.v_bridges[i][j] else 0 for i in range(state.size - 1) for j in range(state.size)]
        
        return state_array
    
    @staticmethod
    def compute_sizes(state: PontuState) -> tuple:
        '''
        Returns the size of the input and output layers.
        Documentation of the reason are write into the code.
        '''
        # Compute the size of the input
        num_pawns = (state.size - 2) * 2 * 2
        num_h_bridges = state.size * (state.size - 1)
        num_v_bridges = (state.size - 1) * state.size
        input_size = num_pawns + num_h_bridges + num_v_bridges

        # Compute the size of the output
        output_pawns = (state.size - 2) + 1  # On node for each pawn and for the None action
        output_directions = 5  # One node for each direction and for the None action
        output_types = 1 # One node for the type of bridge
        output_x = state.size - 1 # One node for each x coordinate
        output_y = state.size # One node for each y coordinate
        output_size = output_pawns + output_directions + output_types + output_x + output_y

        return input_size, output_size
    
    @staticmethod
    def output_to_action(output) -> tuple:
        # Reshape the output
        pawn_id = torch.argmax(output[0:4]).item()
        if pawn_id == 3: return None
        direction = ["EAST", "NORTH", "WEST", "SOUTH", None][torch.argmax(output[4:9]).item()]
        bridge_type = "h" if torch.sigmoid(output[9]).item() > 0.5 else "v"
        bridge_x = torch.argmax(output[10:14]).item()
        bridge_y = torch.argmax(output[14:19]).item()

        # Return the output state
        if bridge_type == "h": return (pawn_id, direction, bridge_type, bridge_x, bridge_y)
        return (pawn_id, direction, bridge_type, bridge_y, bridge_x)
    
    @staticmethod
    def action_to_output(action) -> np.ndarray:
        output = np.zeros(19, dtype=np.int8)
        # Pawns
        if action[0] is None:
            output[3] = 1
        else:
            output[action[0]] = 1
        # Direction
        output[4 + ["EAST", "NORTH", "WEST", "SOUTH", None].index(action[1])] = 1
        # Bridge
        output[9] = 1 if action[2] == "h" else 0
        output[10 + action[3]] = 1
        output[14 + action[4]] = 1
        return output

    @staticmethod
    def action_loss(action1: tuple, action2: tuple) -> int:
        pawn_loss = abs(action1[0] - action2[0])
        direction_loss = int(action1[1] != action2[1])
        bridge_loss = int(action1[2] != action2[2])
        bridge_x_loss = abs(action1[3] - action2[3])
        bridge_y_loss = abs(action1[4] - action2[4])
        return pawn_loss + direction_loss + bridge_loss + bridge_x_loss + bridge_y_loss

####################
# Neuronal Network #
####################
class NeuralNet(nn.Module):
    '''
    Neural network for the Pontu game.
    '''
    def __init__(self, input_size, hidden_size, output_size, number_layers):
        super(NeuralNet, self).__init__()
        # Create the layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(number_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.name = "Neural Network " + ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    def forward(self, x):
        # Connect the layers
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

    def __str__(self):
        return str(self.name)

####################
# Neuronal Genetic #
####################
class NeuralGenetic:
    '''
    Genetic algorithm for the Pontu game.
    '''

    @staticmethod
    def get_action(model: NeuralNet, state: PontuState) -> tuple:
        '''
        Returns the action to do for the given state.
        '''
        # Compute the input and output
        input_features = NeuralHelpers.state_to_features(state)
        input_tensor = torch.from_numpy(input_features).float()
        output_tensor = model(input_tensor)

        # Return the action
        output_action = NeuralHelpers.output_to_action(output_tensor)
        return output_action

    def __init__(self, initial_state: PontuState, population_size: int, nbr_generations: int, mutation_rate: float, mutation_size: float, hidden_size: int = 100, number_layers: int = 1, max_runs: int = 10, max_moves: int = 100, verbose: bool = False) -> None:
        # Genetic attributes
        self.initial_state = initial_state
        self.population_size = population_size
        self.nbr_generations = nbr_generations
        self.mutation_rate = mutation_rate
        self.mutation_size = mutation_size
        self.initial_mutations_rate = mutation_rate
        self.initial_mutations_size = mutation_size

        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.max_runs = max_runs
        self.max_moves = max_moves

        # Neuronal attribues
        self.input_size, self.output_size = NeuralHelpers.compute_sizes(initial_state)

        # Verbos attributes
        self.verbose = verbose
        self.won = 0
        self.loose = 0
        self.invalid_move = 0
        self.nb_runs = 0
        self.turns = 0
 
    def generate_population(self) -> list:
        '''
        Generate a population of neural networks.
        '''
        # Log the generation
        if self.verbose: logging.info("Generating the population {} neural networks.".format(self.population_size))

        # Generate the population
        population = []
        for _ in range(self.population_size):
            population.append(NeuralNet(self.input_size, self.hidden_size, self.output_size, self.number_layers))
        return population       
    
    def mutate(self, model: NeuralNet) -> None:
        '''
        Mutate a neural network.
        '''
        # Mutate the model
        for param in model.parameters():
            if np.random.rand() < self.mutation_rate:
                noise = torch.randn(param.size())
                noise = torch.clamp(noise, -self.mutation_size, self.mutation_size)
                param.data += noise

        """
        for param in model.parameters():
            if np.random.rand() < self.mutation_rate:
                param.data += torch.randn(param.size()) * self.mutation_size
        """
    
    def crossover(self, model1: NeuralNet, model2: NeuralNet) -> NeuralNet:
        '''
        Crossover two neural networks.
        '''
        # Crossover the model
        child = NeuralNet(self.input_size, self.hidden_size, self.output_size, self.number_layers)
        for param1, param2, param_child in zip(model1.parameters(), model2.parameters(), child.parameters()):
            if np.random.rand() < 0.5:
                param_child.data.copy_(param1.data)
            else:
                param_child.data.copy_(param2.data)
        return child
    
    def select(self, population: list, fitness: np.ndarray) -> NeuralNet:
        '''
        Select a model from the population.
        '''
        # Indices of non-zero fitness
        idx = np.nonzero(fitness)[0]

        if len(idx) == 0:
            model_idx = np.random.randint(0, len(population))

        else:
            # Compute the probability of selection based on fitness
            fitness_probability = fitness[idx] / fitness[idx].sum()

            # Select the index of a parent based on its fitness
            model_idx = np.random.choice(idx, p=fitness_probability)

        return population[model_idx],  fitness[model_idx]
    
    def run_game_model(self, model0: NeuralNet, model1: NeuralNet) -> tuple:
        '''
        Returns the result of the game. We assume that we want to compute the fitness of the model0.
        - The first element is the score of the game.
        - The second element is a number that inicates the reason of the end of the game.
            - 0: First player won.
            - 1: Second player won.
            - 2: One player has done an invalid move or one player.
        '''
        verbose = self.verbose
        if verbose: logging.info("Running the game between {} and {}.".format(model0, model1))

        # Play the game
        state = self.initial_state
        while not state.game_over() and state.turns < self.max_moves:
            # Get current player
            current_player = state.cur_player
            action = None

            # Get the action for the current player
            if current_player == 0:
                action = self.get_action(model0, state)
                if verbose: logging.info("Player 0: " + str(action))
                if not state.is_action_valid(action):
                    if verbose: logging.info("Invalide move")
                    return 0, 2, state.turns

            else:
                action = self.get_action(model1, state)
                if verbose: logging.info("Player 1: " + str(action))
                if not state.is_action_valid(action):
                    if verbose: logging.info("Invalide move")
                    return 0.1, 2, state.turns
            
            # Play action
            state.apply_action(action)
        
        # Return th result
        if state.game_over():
            if verbose: logging.info("Game over")
            if state.winner == 0:
                if verbose: logging.info("Plater 0 won")
                return 1, 0, state.turns
        
        if verbose: logging.info("Payer 1 won")
        return 0.5, 1, state.turns
  
    def run_game_agent(self, model: NeuralNet, agent: Agent) -> tuple:
        '''
        Returns the result of the game. We assume that we want to compute the fitness of the model0.
        - The first element is the score of the game.
        - The second element is a number that inicates the reason of the end of the game.
            - 0: First player won.
            - 1: Second player won.
            - 2: One player has done an invalid move or one player.
        '''
        verbose = self.verbose
        if verbose: logging.info("Running the game between {} and angent.".format(model))

        # Prepare the game
        state = self.initial_state.copy()
        action = None
        agent.set_id(1)

        # Plat the game
        while not state.game_over() and state.turns < self.max_moves:
            # Get current player
            current_player = state.cur_player

            # Get the action for the current player
            if current_player == 0:
                action = self.get_action(model, state)
                if verbose: logging.info("Player 0: " + str(action))
                if not state.is_action_valid(action):
                    if verbose: logging.info("Invalide move")
                    return state.turns * 1e-2, 2, state.turns

            else:
                action = agent.get_action(state, action, 10)
                if verbose: logging.info("Player 1: " + str(action))
                if not state.is_action_valid(action):
                    if verbose: logging.info("Invalide move")
                    return state.turns * 1e-2, 2, state.turns
            
            # Play action
            state.apply_action(action)
        
        # Return th result
        if state.game_over():
            if verbose: logging.info("Game over in {} turns".format(state.turns))
            if state.winner == 0:
                if verbose: logging.info("Player 0 won")
                return 1, 0, state.turns
        
        if verbose: logging.info("Payer 1 won")
        return 0.5, 1, state.turns

    def fitness_function(self, model0: NeuralNet, model_or_agent: NeuralNet or Agent) -> float:
        '''
        Return the fitness of the given neural network
        '''
        fitness = 0.0
        for _ in range(self.max_runs):
            self.nb_runs += 1

            # Make the agents play the game
            if isinstance(model_or_agent, NeuralNet):
                fit, msg, turns = self.run_game_model(model0, model_or_agent)
            else:
                fit, msg, turns = self.run_game_agent(model0, model_or_agent)
            
            # Compute statisitc's
            fitness += fit / self.max_runs
            self.turns += turns
            if msg == 0: self.won += 1 
            elif msg == 1: self.loose += 1
            else: self.invalid_move += 1
        return fitness
    
    def envolve(self, population: list, fitnesses: np.ndarray) -> list:
        '''
        Evolve the population.
        '''
        # Create the new population
        new_population = []

        # Keep 5% of the best models
        idx = np.argsort(fitnesses)[::-1]
        for i in range(int(self.population_size * 0.05)):
            new_population.append(population[idx[i]])
        
        # Evolve the rest of the population
        for _ in range(int(self.population_size * 0.95)):
            # Select two parents
            parent1, fit1 = self.select(population, fitnesses)
            parent2, fit2 = self.select(population, fitnesses)

            # Create a child
            child = self.crossover(parent1, parent2)

            # Mutate the child
            # Adapt the mutation rate and the mutation probability according to the fitness
            if min(fit1, fit2) < 0.5 * (self.nb_runs / self.population_size):
                self.mutation_rate = 0.5
                self.mutation_size = 0.5
            else:
                self.mutation_rate = self.initial_mutations_rate
                self.mutation_size = self.initial_mutations_size

            self.mutate(child)

            # Add the child to the new population
            new_population.append(child)

        # Check if the population is valid
        if len(new_population) != self.population_size:
            if self.verbose: logging.info("The population is not valid.")
            # Add or remove models to the population randomly
            if len(new_population) < self.population_size:
                for _ in range(self.population_size - len(new_population)):
                    new_population.append(population[np.random.randint(0, len(population))])
            else:
                for _ in range(len(new_population) - self.population_size):
                    new_population.pop(np.random.randint(0, len(new_population)))

        if self.verbose: logging.info("Evolve the population of size {} (wanted size {}).".format(len(new_population), self.population_size))
        return new_population

    def teacher_forcing(self, model: NeuralNet, input_list: list, input_states: list ,output_matrix: list) -> NeuralNet:
        '''
        Train the model with the teacher forcing method.
        '''
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(model.parameters())

        for i in range(len(input_list)):
            print("Training the model with the teacher forcing method... {}/{}".format(i, len(input_list)), end="\r")
            input_tensor = torch.tensor(input_list[i]).unsqueeze(0)
            output_tensor = torch.tensor(output_matrix[i], dtype=torch.float32).unsqueeze(0)

            optimizer.zero_grad()

            input_features = NeuralHelpers.state_to_features(input_states[i])
            input_tensor = torch.from_numpy(input_features).float()
            output = model(input_tensor)

            loss = criterion(output.unsqueeze(0), output_tensor)

            loss.backward()
            optimizer.step()

        return model

    def pre_train(self, model: NeuralNet, agent: Agent, size: int) -> NeuralNet:
        '''
        Pre-train the model to make it knows the rules of the game.
        '''
        state = self.initial_state
        input_states = [state]

        # Create a list of inputs
        print("Create a list of inputs... ", end="")
        input_list = [NeuralHelpers.state_to_features(state)]
        for _ in range(size):
            state = self.initial_state.copy()
            action = None
            while not state.game_over():
                agent.set_id(state.cur_player)
                action = agent.get_action(state, action, 10)
                state.apply_action(action)
                input_states.append(state)
                input_list.append(NeuralHelpers.state_to_features(state))
        print("Done, list of size {} created.".format(len(input_list)))
        
        # Create a list of outputs
        print("Create a list of outputs... ", end="")
        output_matrix = []
        for i in range(len(input_list)):
            output_actions = input_states[i].get_current_player_actions()
            output_array = []
            for action in output_actions:
                output_array.append(NeuralHelpers.action_to_output(action))
            output_matrix.append(np.array(output_array))
        print("Done, list of size {} created.".format(len(output_matrix)))

        # Train the model
        print("Train the model...")
        model = self.teacher_forcing(model, input_list, input_states, output_matrix)
        return model
    
    def train(self, additional_agent: list, inital_population: NeuralNet = None) -> NeuralNet:
        '''
        Returns the best neural network after training.
        '''
        # Create the initial population
        if inital_population is None:
            population = self.generate_population()
        else:
            population = [inital_population for _ in range(self.population_size)]

        # Train the population
        for generation in range(self.nbr_generations):
            print("Generation {} over {} ".format(generation, self.nbr_generations))
            
            # Compute the fitness of each model
            fitnesses = np.zeros(self.population_size)
            last_invalid_move = 0
            for i in range(self.population_size):
                # Playing against other models if they finished a game at least 50% of the time
                if last_invalid_move > 0.5:
                    for j in range(self.population_size):
                        if i != j:
                            fitnesses[i] += self.fitness_function(population[i], population[j]) 
                # Playing against the additional agent
                for agent in additional_agent:
                    fitnesses[i] += self.fitness_function(population[i], agent)
                
                # Print the progress
                progress = int(50 * i / len(population))
                print("\r[{}{}] {}%".format("=" * progress, " " * (50 - progress), 100 * i / len(population)), end="")

            # Print the result and reset counters
            print("\nWon: {:.2f}%, Loose: {:.2f}%, Invalid move: {:.2f}%, Turns: {:.2f}\n".format(100 * self.won / self.nb_runs, 100 * self.loose / self.nb_runs, 100 * self.invalid_move / self.nb_runs, self.turns / self.nb_runs))
            last_invalid_move = self.invalid_move
            self.won = 0; self.loose = 0; self.invalid_move = 0; self.nb_runs = 0; self.turns = 0
            # Evolve the population
            logging.info("population size: {}, fitnesses size {}.".format(len(population), len(fitnesses)))
            population = self.envolve(population, fitnesses)
        
        # Return the best model
        idx = np.argsort(fitnesses)[::-1]
        return population[idx[0]]
    
    def train_parrallel(self, additional_agent: list, max_threads = 10, inital_population: NeuralNet = None) -> NeuralNet:
        '''
        Returns the best neural network after training. This function uses multiprocessing.
        '''
        # Create the initial population
        if inital_population is None:
            population = self.generate_population()
        else:
            population = [inital_population for _ in range(self.population_size)]

        # Train the population
        for generation in range(self.nbr_generations):
            print("Generation {} over {} ".format(generation, self.nbr_generations))

            # Compute the fitness of each model
            fitnesses = np.zeros(self.population_size)
            last_invalid_move = 0

            # Start the multiprocessing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for i in range(self.population_size):
                    # Playing against other models if they finished a game at least 50% of the time
                    if last_invalid_move > 0.5:
                        for j in range(self.population_size):
                            if i != j:
                                futures.append(executor.submit(self.fitness_function, population[i], population[j]))
                    # Playing against the additional agent
                    for agent in additional_agent:
                        futures.append(executor.submit(self.fitness_function, population[i], agent))
                    
                    # Compute the fitness
                    fitnesses[i] = sum([f.result() for f in futures])
                    futures = []                    
                    # Print the progress
                    progress = int(50 * i / len(population))
                    print("\r[{}{}] {}%".format("=" * progress, " " * (50 - progress), 100 * i / len(population)), end="")

            # Print the result and reset counters
            print("\nWon: {:.2f}%, Loose: {:.2f}%, Invalid move: {:.2f}%, Turns: {}\n".format(100 * self.won / self.nb_runs, 100 * self.loose / self.nb_runs, 100 * self.invalid_move / self.nb_runs, self.turns / self.nb_runs))
            last_invalid_move = self.invalid_move
            self.won = 0; self.loose = 0; self.invalid_move = 0; self.nb_runs = 0; self.turns = 0
            # Evolve the population
            logging.info("population size: {}, fitnesses size {}.".format(len(population), len(fitnesses)))
            population = self.envolve(population, fitnesses)
        
        # Return the best model
        idx = np.argsort(fitnesses)[::-1]
        return population[idx[0]]

    

if __name__ == "__main__":
    initial_state = PontuState()

    # Genetic algorithm parameters
    population_size = 1000
    nbr_generations = 100

    mutation_rate = 0.5
    mutation_size = 0.5

    hidden_size = 100
    number_layers = 2

    max_runs = 10
    max_moves = 200
    verbose = True

    # Create the genetic algorithm
    genetic_algorithm = NeuralGenetic(initial_state, population_size, nbr_generations, mutation_rate, 
                                      mutation_size, hidden_size, number_layers, max_runs, max_moves, verbose)

    # Create the additional agent
    additional_agents = [RandomAgent(), BasicAgent()]

    # Get the pre-trained model
    # trained_model = NeuralNet(52, hidden_size, 19, number_layers)
    # state_dict = torch.load('pre_trained.pt', map_location=torch.device('cpu'))
    # trained_model.load_state_dict(state_dict)
    # trained_model.eval()

    # Create the population
    model0 = genetic_algorithm.generate_population()[0]


    # Train the model
    model = genetic_algorithm.pre_train(model0, additional_agents[1], 1000)
    torch.save(model.state_dict(), 'pre_trained.pt')
    
    # model = genetic_algorithm.train(additional_agents)
    model = genetic_algorithm.train_parrallel(additional_agents, max_threads=10, inital_population=model)
    torch.save(model.state_dict(), 'trained.pt')
