import weighted_agent as agent
import numpy as np
from pontu_tools import *
import concurrent.futures
import queue

####################
# Training Genetic #
####################
class NeuralGenetic:

    def __init__(self, weight_len, myagent):
        self.weight_len = weight_len
        self.myagent = myagent

    def generate_weights(self) -> list:
        return np.random.randint(1, high=100, size=self.weight_len, dtype=int)
    
    def generate_population(self, size) -> list:
        population = []
        for _ in range(size):
            population.append(self.generate_weights())
        return population
    
    def mutate(self, weights:list, mutation_rate=0.1) -> list:
        for i in range(len(weights)):
            if np.random.rand() < mutation_rate:
                weights[i] = np.random.randint(1, high=100, dtype=int)
        return weights

    def crossover(self, weights1:list, weights2:list) -> list:
        child = []
        for i in range(len(weights1)):
            if np.random.rand() < 0.5:
                child.append(weights1[i])
            else:
                child.append(weights2[i])
        return child
    
    def select(self, population:list, fitness:np.ndarray, k=2) -> list:
        parents = []
        for _ in range(k):
            idx = np.random.choice(len(population), p=fitness)
            parents.append(population[idx])
        return parents
    
    def fitness(self, weight:list, opponents:list, games=5) -> float:
        # Game variables
        initial_state = PontuState()
        time_out = 100
        display_gui = False
        verbosity = 0

        # Create agent
        fitness = 0
        for opponent in opponents:  
            for _ in range(games):
                # Print a brogress bar with the progress percentage at the end
                # print('\rProgress: [{0:50s}] {1:.1f}%'.format('=' * int((fitness)/(games*len(opponents))*50), (fitness)/(games*len(opponents))*100), end='\r', flush=True)
                # My agent
                agent0 = getattr(__import__(self.myagent), 'MyAgent')()
                agent0.set_weights(weight)
                agent0.set_id(0)

                # Opponent agent
                agent1 = getattr(__import__(opponent), 'MyAgent')()
                agent1.set_id(1)
                res = play_game(initial_state.copy(), [agent0.get_name(), agent1.get_name()], [agent0, agent1], time_out, display_gui, verbosity)
                if res[0] == 0:
                    fitness += 1
                elif res[0] == -1:
                    fitness += 0.5
        return fitness
    
    def evolve(self, population:list, fitness:np.ndarray, mutation_rate=0.1, k=2) -> list:
        new_population = []
        for _ in range(len(population)):
            parents = self.select(population, fitness, k)
            child = self.crossover(parents[0], parents[1])
            child = self.mutate(child, mutation_rate)
            new_population.append(child)
        return new_population
    
    def run(self, population_size=5, mutation_rate=0.1, k=2, generations=10, opponents=['random_agent', 'basic_agent', 'better_agent_1', 'better_agent_3']):
        population = self.generate_population(population_size)
        for i in range(generations):
            # Print a brogress bar with the progress percentage at the end

            # Compute fitness
            fitness = np.zeros(population_size)
            for j in range(population_size):
                # Print a brogress the number of math done
                print("Computing fitness: {}/{}".format(j, population_size * generations))

                fitness[j] = self.fitness(population[j], opponents)
            
            # Normalize fitness
            if np.abs(np.sum(fitness)) < 0.1:
                # handle the case where the range is zero (e.g. set all fitness values to 0)
                fitness = np.array([1/population_size for _ in range(population_size)])
            else:
                fitness = fitness / np.sum(fitness)
            if np.sum(fitness) != 0:
                diff = 1 - np.sum(fitness)
                fitness = fitness + diff / population_size
            
            print('\nFitness:', fitness)

            # Print best weights
            best_agent = population[np.argmax(fitness)]
            print('Best agent weights:', best_agent, '\n')
            print('Best agent fitness:', np.max(fitness))
            print('Average fitness:', np.mean(fitness))
            print('Worst fitness:', np.min(fitness))

            # Evolve population
            population = self.evolve(population, fitness, mutation_rate, k)

        # Get best agent
        fitness = np.array([self.fitness(agent, opponents) for agent in population])
        best_agent = population[np.argmax(fitness)]

        # Print the results
        print('Best agent weights:', best_agent)
        print('Best agent fitness:', np.max(fitness))
        print('Average fitness:', np.mean(fitness))
        print('Worst fitness:', np.min(fitness))

if __name__ == '__main__':
    genetic = NeuralGenetic(7, 'weighted_agent')
    genetic.run(population_size=5, mutation_rate=0.1, k=2, generations=10, opponents=['random_agent', 'basic_agent', 'better_agent_1', 'better_agent_3'])