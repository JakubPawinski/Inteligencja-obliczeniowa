import pandas as pd
import numpy
import pygad
import math

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

def fitness_function(pyGad_class, solution, solution_idx):
    x, y, z, u, v, w = solution
    fitness = endurance(x, y, z, u, v, w)
    return fitness

gene_space = {'low': 0, 'high': 1}
sol_per_pop = 10
num_genes = 6


num_parents_mating = 5
num_generations = 30
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 15

num_runs = 10

best_solutions = []

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")
    ga_instance = pygad.GA(gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes)
    
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    best_solutions.append(solution_fitness)

print("Best solutions from all runs:", max(best_solutions))
ga_instance.plot_fitness()