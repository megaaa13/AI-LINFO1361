import numpy as np
import signal
import time
import pygame
import traceback

# Pontu
from pontu_state import PontuState
from pontu_gui import GUIState
from pontu_tools import handle_timeout, TimerDisplay
from agent import Agent

# PyTorch
import torch
import torch.nn as nn
  

##################
# Neural Helpers #
##################
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
        output_pawns = (state.size - 2)  # On node for each pawn
        output_directions = 4  # One node for each direction
        output_types = 1 # One node for the type of bridge
        output_x = state.size - 1 # One node for each x coordinate
        output_y = state.size # One node for each y coordinate
        output_size = output_pawns + output_directions + output_types + output_x + output_y

        return input_size, output_size


##################
#   Neural Net   #
##################
class NeuralNet(nn.Module):
    '''
    Neural network for the Pontu game.
    '''
    def __init__(self, intput_size, hidden_size, output_size):
        print("Initializing the neural network")
        print("Input size: ", intput_size)
        print("Hidden size: ", hidden_size)
        print("Output size: ", output_size)
        
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(intput_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Connect the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape the output
        pawn_id = torch.argmax(x[0:3]).item()
        direction = ["EAST", "NORTH", "WEST", "SOUTH"][torch.argmax(x[3:7]).item()]
        bridge_type = "h" if torch.sigmoid(x[7]).item() > 0.5 else "v"
        bridge_x = torch.argmax(x[8:12]).item()
        bridge_y = torch.argmax(x[12:17]).item()

        # Return the output state
        if bridge_type == "h": return (pawn_id, direction, bridge_type, bridge_x, bridge_y)
        return (pawn_id, direction, bridge_type, bridge_y, bridge_x)
    

##################
#  Neural  Game  #
##################
class NeuralGenetic:
    '''
    Genetic algorithm for the Pontu game.
    '''
    @staticmethod
    def get_action(model: NeuralNet, state: PontuState) -> tuple:
        '''
        Returns the action corresponding to the output.
        '''
        # Get the input data and get the output action
        input_data = NeuralHelpers.state_to_features(state)
        input_tensor = torch.from_numpy(input_data).float()
        output_action = model(input_tensor)
        return output_action
    
    @staticmethod
    def generate_population(population_size: int, input_size: int, hidden_size: int, output_size: int) -> list:
        '''
        Returns a list of neural networks.
        '''
        return [NeuralNet(input_size, hidden_size, output_size) for _ in range(population_size)]
    
    @staticmethod
    def mutate(model: NeuralNet, mutation_rate: int, mutation_size: int) -> None:
        '''
        Mutates the neural network with a given mutation rate.
        '''
        for param in model.parameters():
            if torch.rand(1) < mutation_rate:
                param.data += torch.randn(param.data.size()) * mutation_size
    
    @staticmethod
    def crossover(parent1: NeuralNet, parent2: NeuralNet) -> NeuralNet:
        '''
        Returns a new neural network that is the crossover of the two parents.
        It is a simple crossover that takes the parameters of the parents randomly.
        '''
        child = NeuralNet(parent1.fc1.in_features, parent1.fc1.out_features, parent1.fc3.out_features)
        for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            if torch.rand(1) < 0.5:
                child_param.data.copy_(parent1_param.data)
            else:
                child_param.data.copy_(parent2_param.data)
        return child
    
    @staticmethod
    def run_game(model0: NeuralNet, model1:NeuralNet, state: PontuState, max_iter:int) -> tuple:
        '''
        Returns the result of the game.
        - The first element is the score of the game.
        - The second element is a number that inicates the reason of the end of the game.
            - 0: The game is over and the first player won.
            - 1: The game is over and the second player won.
            - 2: The game is over and the first player has done an invalid move.
            - 3: The game is over and the second player has done an invalid move.
            - 4: The game is over and player has no more moves.
        '''

        # Play the game
        while not state.game_over() and state.turns < max_iter:
            # Get current player
            current_player = state.cur_player
            
            # Get the action for the current player
            if current_player == 0:
                action = NeuralGenetic.get_action(model0, state)
                if not state.is_action_valid(action): return 0, 2
            else:
                action = NeuralGenetic.get_action(model1, state)
                if not state.is_action_valid(action): return 0.1, 3

            # Play the action
            state.apply_action(action)
        
        # Return the result
        if state.game_over():
            if state.winner == 0: return 1, 0
            return 0.3, 1
        return 0.5, 4
        
    
    @staticmethod
    def fitness_function(model1: NeuralNet, model2: NeuralNet, state: PontuState, max_iter:int, nbr_runs: int) -> tuple:
        '''
        Returns the fitness of the given neural network.
        '''
        fitness = 0.0
        winnings = 0
        for _ in range(nbr_runs):
            result, won = NeuralGenetic.run_game(model1, model2, state, max_iter)
            if won: winnings += 1
            fitness += result
        return fitness / nbr_runs, winnings / nbr_runs
    
    @staticmethod
    def select_parents(population: list, fitnesses: np.ndarray) -> NeuralNet:
        '''
        Returns a parent for mating using the roulette wheel selection.
        '''
        fitnesses_prob = fitnesses / np.sum(fitnesses)
        try:
            parent_idx = np.random.choice(len(population), p=fitnesses_prob)
        except:
            parent_idx = np.random.choice(len(population))
        return population[parent_idx]

    @staticmethod
    def evolve(population: list, fitnesses: np.ndarray, mutation_rate: int, mutation_size: int) -> list:
        '''
        Returns a new population after applying the genetic algorithm.
        '''
        new_population = []
        for _ in range(len(population)):
            parent1 = NeuralGenetic.select_parents(population, fitnesses)
            parent2 = NeuralGenetic.select_parents(population, fitnesses)
            child = NeuralGenetic.crossover(parent1, parent2)
            NeuralGenetic.mutate(child, mutation_rate, mutation_size)
            new_population.append(child)
        return new_population
    
    @staticmethod
    def train(population_size: int, hidden_size: int, mutation_rate: int, mutation_size: int, nbr_generations: int, state: PontuState, max_iter: int, nbr_runs: int=10) -> NeuralNet:
        '''
        Returns the best neural network after training.
        '''
        # Create the initial population
        input_size, output_size = NeuralHelpers.compute_sizes(state)
        population = NeuralGenetic.generate_population(population_size, input_size, hidden_size, output_size)

        # Train the population
        for generation in range(nbr_generations):
            print("Generation: {} over {}".format(generation, nbr_generations))
            fitnesses = []
            woning_glob = 0
            losing_glob = 0
            invalid_moves_glob = 0
            no_more_moves_glob = 0

            for i in range(len(population)):
                fitness = 0.0
                woning = 0
                losing = 0
                invalid_moves = 0
                no_more_moves = 0
                for j in range(len(population)):
                    # Skip if the models are the same
                    if i == j: continue
                    # Get the fitness of the models by having them play against each other
                    fitness_play, won = NeuralGenetic.fitness_function(population[i], population[j], state, max_iter, nbr_runs)
                    fitness += fitness_play
                    if won == 0: woning += 1
                    elif won == 1: losing += 1
                    elif won == 2: invalid_moves += 1
                    elif won == 3: invalid_moves += 1
                    elif won == 4: no_more_moves += 1
                                        
                fitnesses.append(fitness)
                woning_glob += woning / (len(population) - 1)
                losing_glob += losing / (len(population) - 1)
                invalid_moves_glob += invalid_moves / (len(population) - 1)
                no_more_moves_glob += no_more_moves / (len(population) - 1)
                
                # Print a progress bar
                progress = int(50 * len(fitnesses) / len(population))
                print("\r[{}{}] {}%".format("=" * progress, " " * (50 - progress), 100 * len(fitnesses) / len(population)), end="")
            print("\nGeneration {}: \n\tWonning times: {}\n\tBest fitness: {}\n\tLosing times: {}\n\tInvalid moves: {}\n\tNo more moves: {}".format(generation, woning_glob / len(population), max(fitnesses), losing_glob / len(population), invalid_moves_glob / len(population), no_more_moves_glob / len(population)))
            population = NeuralGenetic.evolve(population, np.array(fitnesses), mutation_rate, mutation_size)

        # Determine the best model
        best_model = None
        best_fitness = -np.inf
        for model, fitness in zip(population, fitnesses):
            if fitness > best_fitness:
                best_model = model
                best_fitness = fitness
        return best_model


##################
#  Pontu modify  #
##################
class PontuModify:

    @staticmethod
    def get_action_timed(player, state, last_action, time_left):
        '''
        Returns the action of the player given the state and the time left.
        This method differs from the get_action_timed() method in the pontu_tools.py file
        by the fact that it use the NeuralGenetic.get_action() method instead of the
        player.get_action() method for the neural network.
        '''
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, time_left)
        exe_time = time.time()
        try:
            if isinstance(player, Agent):
                action = player.get_action(state, last_action, time_left)
            else:
                action = NeuralGenetic.get_action(player, state)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            exe_time = time.time() - exe_time
        return action, exe_time
    
    @staticmethod
    def run_game(players: tuple, init_state: PontuState, total_time, display_gui: bool) -> tuple:
        '''
        Plays a game with the given neural network and returns the score and if the model has won the game or not.
        '''
        # Initialize the variables
        state = init_state
        time_left = [total_time for _ in range(state.size - 2)]
        timedout = -1
        crashed = -1
        invalidaction = -1
        quit = -1
        action = None
        last_action = None
        if display_gui:
            gui = GUIState()

        # Play the game
        while not state.game_over():
            cur_player = state.get_cur_player()
            timer_stop = [False]
            if display_gui:
                timer = TimerDisplay(gui, cur_player, time_left.copy(), timer_stop)
                gui.display_state(state)
                pygame.display.flip()
                timer.start()
            try:
                action, exe_time = PontuModify.get_action(players[cur_player], state, last_action, time_left[cur_player])
            except TimeoutError:
                # Current player timed out
                timedout = cur_player
                state.set_timed_out(cur_player)
                break
            except Exception as e:
                trace = traceback.format_exc().split('\n')
                exception = trace[len(trace) - 2]
                # Current player crashed
                crashed = cur_player
                break
            else:
                # update time
                timer_stop[0] = True
                time_left[cur_player] -= exe_time
                # check if the action is valid
                try:
                    if action[0] == 'rage-quit':
                        # the player wants to quit
                        quit = cur_player
                        state.winner = 1 - quit
                        break
                    elif state.is_action_valid(action):
                        # The action is valid so we can apply the action to the state
                        state.apply_action(action)
                        last_action = action
                    else:
                        # Set that the current player gave an invalid action
                        invalidaction = cur_player
                        state.set_invalid_action(invalidaction)
                        break
                except Exception:
                    # set that the current player gave an invalid action
                    invalidaction = cur_player
                    state.set_invalid_action(cur_player)
                    break
            if display_gui:
                timer.join()
        if display_gui:
            gui.display_winner(state)

        # Return the score and if the model has won the game or not
        if timedout != -1:
            if isinstance(players[timedout], NeuralNet):
                return (0, False)
            else:
                print('Timed out')
                return (0.25, True)
        elif crashed != -1:
            if isinstance(players[crashed], NeuralNet): 
                return (0, False)
            else: 
                print('Crashed')
                return (0.25, True)
        elif invalidaction != -1:
            if isinstance(players[invalidaction], NeuralNet):
                return (0, False)
            else: 
                print('Invalid action')
                return (0.25, True)
        elif quit != -1:
            if isinstance(players[quit], NeuralNet): return (0, False)
            else: 
                print('Quit')
                return (0.25, True)
        else:
            if isinstance(players[state.get_winner()], NeuralNet): return (1, True)
            else: return (0.5, True)   


        '''
        best_model = population[0]
        best_fitness = NeuralGenetic.fitness_function(best_model, player, state, total_time, display_gui, nbr_runs)
        for model in population:
            fitness = NeuralGenetic.fitness_function(model, player, state, total_time, display_gui, nbr_runs)
            if fitness > best_fitness:
                best_model = model
                best_fitness = fitness
        '''   

# Main program
if __name__ == "__main__":
    initial_state = PontuState()

    population_size = 100
    hidden_size = 100
    mutation_rate = 0.1
    mutation_size = 0.1
    nbr_generations = 100
    max_iter = 200
    nbr_runs = 10

    best_model = NeuralGenetic.train(population_size, hidden_size, mutation_rate, mutation_size, 
                                     nbr_generations, initial_state, max_iter, nbr_runs)
    
    # Save the best model
    # torch.save(best_model, 'best_model.pt')
