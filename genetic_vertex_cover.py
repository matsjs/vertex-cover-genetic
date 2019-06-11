import numpy as np
import time
import random
random.seed(21) # Magic number

edges = []
num_verts = 0

def input_data(filename):
    global edges
    global num_verts
    line_num = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if len(line) > 0 and line[0] != 'c':
                if line[0] == 'p':
                    num_verts = int(line.split()[2])
                    edges = np.zeros((int(line.split()[3]), 2), int)
                else:
                    np.put(edges, (line_num*2, line_num*2+1), (int(line.split()[0])-1, int(line.split()[1])-1))
                    line_num += 1

class Agent:
    # Initialization of agent
    def __init__(self, generation, dna_a, dna_b, mutation_chance=5):
        self.generation = generation
        self.dna = np.zeros(len(dna_a), int)
        # Chance that gene will mutate. Initially high rate, then quickly reducing.
        self.mutation_rate = mutation_chance - (mutation_chance/(1+np.exp(-0.1*(generation-np.sqrt(len(dna_a))))))
        self.genesis(dna_a, dna_b)


    # Changes genetic make-up 
    def mutate(self):
        self.dna = [self.flip(gene) if (random.randint(0,100) < self.mutation_rate) else gene for gene in self.dna]
        return self
    
    def flip(self, num):
        if num == 0:
            return 1
        else:
            return 0

    # Breeds this individual with another creating offspring
    def breed(self, other, offspring_number):
        return [Agent(self.generation + 1, self.dna, other.dna) for i in range(offspring_number)]
        
    # Creates a wholly new genetic make-up at random
    def genesis(self, dna_a, dna_b, selection_percent = 50):
        self.dna = [dna_a[i] if (random.randint(0,100) < selection_percent) else dna_b[i] for i in range(len(dna_a))]
        self.fitness = self.set_fitness()

    # Returns fitness of agent
    def set_fitness(self):
        vert_cover = []
        for edge in edges:
            if self.dna[edge[0]] and self.dna[edge[1]]:
                vert_cover.append(0.5)
            elif self.dna[edge[0]] or self.dna[edge[1]]:
                vert_cover.append(1)
            else:
                vert_cover.append(-2)
        # vert_cover = [1 if (self.dna[edge[0]] == 1 or self.dna[edge[1]] == 1) else -1 for edge in edges]
        return np.sum(vert_cover)
        # num_chosen_verts = np.sum(self.dna)
        # if num_chosen_verts == 0:
        #     return 0
        # else:
        #     return (np.sum(vert_cover)**2/np.sum(self.dna))

    def __str__(self):
        return ('Generation: {} | Fitness: {}'.format(self.generation,self.fitness))


class Genetic_vertex_cover:
    # Selects solution method and returns agents
    def solve(self, filename, exit_condition='convergence', max_generations=100, num_agents=10):
        """ Set up problem space

        Keyword arguments:
        value_weights -- [(int, int),...,(int, int)] 
        knapsack_size -- int
        exit_condition -- 'convergence', 'max_gen'
        """
        print("Loading data..")
        input_data(filename)
        self.max_val_gen = []
        self.num_agents = num_agents
        self.first_generation = [Agent(1, np.ones(num_verts), np.zeros(num_verts)) for i in range(self.num_agents)]
        self.max_val_gen.append(np.max([agent.fitness for agent in self.first_generation]))
        result = 0
        if exit_condition == 'convergence':
            result= self.converge_solution(self.first_generation, max_generations)
        elif exit_condition == 'max_gen'  and max_generations > 0:
            result= self.gen_cap_solution(self.first_generation, max_generations)

        result.sort(key=lambda ag: ag.fitness, reverse=True)
        return result[0]


    def converge_solution(self, agents, max_generations):
        ''' Solution by convergence of max fitness. Stops at max_generations if convergence has not been reached.
        '''
        curr_gen = 0
        delta_fitness = 2

        while delta_fitness > 0.1 and curr_gen < max_generations: # Convergence when best agent stops improving
            winners = self.select_winners(agents) # Select winners and lucky losers
            next_gen = self.breed(winners) # Breed next generation
            mut_gen = [agent.mutate() for agent in next_gen] # Mutate generations
            current_fitness = np.max([agent.fitness for agent in mut_gen]) # Calculate fitness
            if len(self.max_val_gen) >= 3:
                delta_fitness = np.mean(abs(np.diff(self.max_val_gen[-3:])))
            agents = mut_gen
            mut_rate = agents[0].mutation_rate
            curr_gen += 1
            self.max_val_gen.append(current_fitness)
            print(f"Generation {curr_gen}, delta {delta_fitness}, mutation rate {mut_rate}")
        return agents


    def gen_cap_solution(self, agents, max_gen):
        ''' Breeds a given number of generations, irrespective of fitness.
        '''
        for i in range(max_gen):
            winners = self.select_winners(agents)
            next_gen = self.breed(winners)
            mut_gen = [agent.mutate() for agent in next_gen]
            agents = mut_gen
            mut_rate = agents[0].mutation_rate
            self.max_val_gen.append(np.max([agent.fitness for agent in agents]))
            print(f"Generation {i}/{max_gen}, mutation rate {mut_rate}")
        return agents

    def select_winners(self, agents):
        ''' Select top 20% by fitness and 5% randomly selected lucky losers
        '''
        top_n = round(self.num_agents * 0.2)
        lucky_losers = round(self.num_agents * 0.05)
        winners = sorted(agents, key=lambda ag: ag.fitness, reverse=True)
        return winners[:top_n] + random.sample(winners[top_n:], lucky_losers)

    def breed(self, breeders):
        ''' Breed agents
        '''
        offspring = round( self.num_agents / (len(breeders)/2))
        random.shuffle(breeders)
        children = []
        for i in range(0,len(breeders)//2):
            children += breeders[i*2].breed(breeders[i*2+1], offspring)
        return children


"""
Problem space initialization
"""
# def run_timetests(tests):
#     results = [['Dynamic Programming', 'Greedy Algorithm', 'Genetic Algorithm']]
#     for test in tests:
#         functions = [knapsack_dp, knapsack_greedy, Genetic_knapsack()]
#         test_results = []
#         for func in functions:
#             start = time.time()
#             try:
#                 result = func.solve(test[0], test[1])                
#             except (MemoryError, RecursionError) as re:
#                 result = None
#             finally:
#                 end = time.time()
#                 test_results.append((result, (end-start)/100))
#         results.append(test_results)

#     return results

# './data/vc-exact_001.gr'
def plot_fitness():
    test = Genetic_vertex_cover()
    start = time.time()
    final_gen = test.solve('./data/vc-exact_005.gr', max_generations=100, num_agents=10, exit_condition='convergence')
    end = time.time()
    print("Time taken: ", end-start)
    print('Fitness: ', final_gen.fitness)
    # print(final_gen.dna)
    from matplotlib import pyplot as plt
    plt.plot([i for i in range(len(test.max_val_gen))],test.max_val_gen)
    plt.show()


plot_fitness()