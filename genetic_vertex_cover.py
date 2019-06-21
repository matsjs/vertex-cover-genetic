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
      vert_cover_fitness2 = np.full([num_edges], -num_edges)
      mask = (self.dna[edges[:, 0]] | self.dna[edges[:, 1]]).astype(bool)
      vert_cover_fitness2[mask] = 1.0

      self.fitness = np.sum(vert_cover_fitness2) - np.sum(self.dna)
      return self.fitness

    # Changes genetic make-up 
    def mutate(self):
      self.dna = self.dna.copy()
      mutation_mask = np.random.rand(num_verts)<self.mutation_rate/num_verts
      self.dna[mutation_mask] = 1 - self.dna[mutation_mask]
      return self
    
    # Breeds this individual with another creating offspring
    def breed(self, other, offspring_number):
      selection = np.random.choice([0, 1], size=(num_verts)).astype(np.bool)
      dna = np.choose(selection, [self.dna, other.dna])
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
        
        initial_dna = np.ones(num_verts, dtype=np.int_) #dict([(x,1) for x in range(num_verts)]) #
        first_generation = [Agent(1, initial_dna , mutation_rate) for i in range(self.num_agents)] 
        
        performance = [agent.update_fitness() for agent in first_generation] 
        
        self.fitness_scores.append(np.max(performance))
        
        result = self.converge_solution(first_generation)
        
        result.sort(key=lambda ag: ag.fitness, reverse=True)
        return result[0]

      
    def converge_solution(self, agents):
        ''' Solution by convergence of max fitness. Stops at max_generations if convergence has not been reached.
        '''
        curr_gen = 0
        delta_fitness = 2
        start_time = time.time()
        
        selection_time = 0
        breed_time = 0
        mutation_time = 0
        fitness_time = 0

        while delta_fitness > 1 and time.time() - start_time < 59 : # Convergence when best agent stops improving
            strt = time.time()
            winners = self.select_winners(agents) # Select winners and lucky losers
            selection_time += time.time() - strt
            
            strt = time.time()
            next_gen = self.breed(winners) # Breed next generation
            breed_time += time.time() - strt
            
            strt = time.time()
            mut_gen = [agent.mutate() for agent in next_gen] # Mutate generations
            mutation_time += time.time() - strt

            strt = time.time()
            performance = [agent.update_fitness() for agent in mut_gen] # Update fitness
            fitness_time += time.time() - strt
            
            if len(self.fitness_scores) >= 50:
                delta_fitness = abs(np.sum(np.diff(self.fitness_scores[-50:])))
            agents = mut_gen

            curr_gen += 1
            self.fitness_scores.append( np.max(performance) )
        
        times = np.sum([selection_time,breed_time,mutation_time,fitness_time])
#         print(f'Selection: {round(selection_time*100/times,2)}, Breeding: {round(breed_time*100/times,2)}, Mutation: {round(100*mutation_time/times,2)}, Fitness: {round(100*fitness_time/times,2)}')
        
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
      
    def create_mut_table(self, mutation_rate):
      return [sum(np.random.binomial(num_verts, mutation_rate/num_verts, 20000) == i)/20000 for i in range(10)]


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


def plot_fitness():
    test = Genetic_vertex_cover()
    start = time.time()
    final_gen = test.solve('./data/vc-exact_031.gr', num_agents= 50, exit_condition= 'convergence', mutation_rate = 1)
    end = time.time()
    print("Time taken: ", end-start)
    print('Fitness: ', final_gen.fitness)
    from matplotlib import pyplot as plt
    plt.plot([i for i in range(len(test.fitness_scores))],test.fitness_scores)
    plt.show()
plot_fitness()