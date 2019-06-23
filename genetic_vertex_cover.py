import numpy as np
import time
import random
import os

def input_data(filename):
    global edges
    global num_verts
    global num_edges
    edges = np.array([])
    num_verts = 0
    num_edges = 0
    line_num = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if len(line) > 0 and line[0] != 'c':
                if line[0] == 'p':
                    num_verts = int(line.split()[2])
                    num_edges = int(line.split()[3])
                    edges = np.zeros((num_edges, 2),dtype=np.int_)
                else:
                    np.put(edges, (line_num*2, line_num*2+1), (int(line.split()[0])-1, int(line.split()[1])-1))
                    line_num += 1

class Agent:
    # Initialization of agent
    def __init__(self, generation, dna, mutation_rate): 
        self.generation = generation
        self.dna = dna
        self.mutation_rate = mutation_rate
        self.fitness = 0
    
    # Returns fitness of agent
    def update_fitness(self):
        # Sum up the number of vertices, and subtract the number of edges not covered * number of edges
        vert_cover_fitness = np.full([num_edges], -num_edges)
        mask = (self.dna[edges[:, 0]] | self.dna[edges[:, 1]]).astype(bool)
        vert_cover_fitness[mask] = 1.0
        # Subtract the number of vertices in cover
        self.fitness = np.sum(vert_cover_fitness) - np.sum(self.dna)
        return self.fitness

    # Changes genetic make-up, flip some number of random genes
    def mutate(self):
      self.dna = self.dna.copy() # Fixes memory issue
      mutation_mask = np.random.rand(num_verts)<self.mutation_rate/num_verts
      self.dna[mutation_mask] = 1 - self.dna[mutation_mask] # Flip genes
      return self
    
    # Breeds this individual with another creating offspring
    def breed(self, other, offspring_number):
      selection = np.random.choice([0, 1], size=(num_verts)).astype(np.bool)
      dna = np.choose(selection, [self.dna, other.dna]) # Uniform selection
      return [Agent(self.generation + 1,
                      dna,
                      self.mutation_rate)
                      for i in range(offspring_number)]


class Genetic_vertex_cover:

    # Selects solution method and returns agents
    def solve(self, filename, exit_condition='convergence', num_agents=50, mutation_rate=1):
        """ Set up problem space
        """
        input_data(filename)

        self.fitness_scores = []
        self.num_agents = num_agents
        
        initial_dna = np.ones(num_verts, dtype=np.int_) 
        first_generation = [Agent(1, initial_dna , mutation_rate) for i in range(self.num_agents)] 
        
        performance = [agent.update_fitness() for agent in first_generation] 
        
        self.fitness_scores.append(np.max(performance))
        
        result = self.converge_solution(first_generation)
        
        result.sort(key=lambda ag: ag.fitness, reverse=True)
        return result[0]

      
    def converge_solution(self, agents):
        ''' Solution by convergence. Time-outs if convergence has not been reached within 59s.
        '''
        curr_gen = 0
        delta_fitness = 2

        start_time = time.time()

        while delta_fitness > 1 and time.time() - start_time < 59 : # Convergence when best agent stops improving
            winners = self.select_winners(agents) # Select winners and lucky losers
            
            next_gen = self.breed(winners) # Breed next generation
            
            mut_gen = [agent.mutate() for agent in next_gen] # Mutate generations

            performance = [agent.update_fitness() for agent in mut_gen] # Update fitness
            
            if len(self.fitness_scores) >= 50: # Start calculating delta fitness at 50 generations
                delta_fitness = abs(np.sum(np.diff(self.fitness_scores[-50:])))
            agents = mut_gen

            curr_gen += 1
            self.fitness_scores.append( np.max(performance) )
        
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
      


# Test GA on benchmark examples
def benchmark():
    test = Genetic_vertex_cover()
    seeds = [i for i in range(10)]

    for dataset in sorted(os.listdir('./data')):
        generations = []
        fitness = []
        deltas = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            start = time.time()
            output = test.solve('./data/' + dataset, num_agents= 50, exit_condition= 'convergence', mutation_rate = 1)
            generations.append(output.generation)
            fitness.append(output.fitness)
            deltas.append(time.time() - start)
        print(f"Dataset: {dataset} | Vertices/ Edges {num_verts}/{num_edges} | Num agents {50} | Time {np.mean(deltas)} | Generations {np.mean(generations)} | Fitness {np.mean(fitness)}/Fitness {np.max(fitness)}") 

# Plot fitness of a run
def plot_fitness(filepath):
    test = Genetic_vertex_cover()
    start = time.time()
    final_gen = test.solve(filepath, num_agents= 50, exit_condition= 'convergence', mutation_rate = 1)
    end = time.time()
    print("Time taken: ", end-start)
    print('Fitness: ', final_gen.fitness)
    from matplotlib import pyplot as plt
    plt.plot([i for i in range(len(test.fitness_scores))],test.fitness_scores)
    plt.show()

# benchmark()
# plot_fitness('./data/vc-exact_031.gr')