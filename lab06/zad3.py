import pandas as pd
import numpy
import pygad
import time
maze = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', 'S', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '1', '0', '1', '1', '1', '0', '1'],
    ['1', '0', '1', '0', '0', '0', '1', '0', '0', '0', '1'],
    ['1', '0', '1', '1', '1', '1', '1', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '0', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '1', '1', '0', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', 'E'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

gene_space = [0, 1, 2, 3]
sol_per_pop = 100
num_genes = 30
num_parents_mating = 10

def fitness_function(pyGad_class, solution, solution_idx):
    start_cords = (1, 1) # Starting position (S)
    end_cords = (9, 10) # Ending position (E)

    # Decode the solution
    path = []
    x, y = start_cords[0], start_cords[1] # Starting position (S)
    for i in solution:
        if i == 0: # Move up
            x -= 1
        elif i == 1: # Move down
            x += 1
        elif i == 2: # Move left
            y -= 1
        elif i == 3: # Move right
            y += 1

        # Check if the move is valid
        if x < 0 or x >= len(maze) or y < 0 or y >= len(maze[0]):
            break
        if maze[x][y] == '1':
            break
        elif maze[x][y] == 'E':
            path.append((x, y))
            break
        else:
            path.append((x, y))

    # Check if the path is valid
    if len(path) == 0:
        return -1
    
    if maze[path[-1][0]][path[-1][1]] == 'E':
        fitness = 1000 - len(path)
        return fitness
    else:
        distance = abs(path[-1][0] - end_cords[0]) + abs(path[-1][1] - end_cords[1])
        fitness = 500 - (distance * 10) - len(path)
        return max(1, fitness)  


keep_parents = 2
num_generations = 100

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10


num_runs = 10
target_fitness = 970
successful_times = []
for run in range(num_runs):
    start = time.time()
    print(f"\nRun {run+1}/{num_runs}")
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
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=[f"reach_{target_fitness}"])
        
    ga_instance.run()
    end = time.time()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Time taken: ", end - start, "seconds")

    if solution_fitness >= target_fitness:
        successful_times.append(end - start)

# print(successful_times)
print("\nAverage time of successful runs: ", numpy.mean(successful_times), "seconds")
ga_instance.plot_fitness()



def visualize_path(solution):
    maze_copy = [row[:] for row in maze]
    
    x, y = 1, 1  
    for i in solution:
        if i == 0: x -= 1
        elif i == 1: x += 1
        elif i == 2: y -= 1
        elif i == 3: y += 1
        
        if x < 0 or x >= len(maze_copy) or y < 0 or y >= len(maze_copy[0]):
            break
        
        if maze_copy[x][y] == '1':
            break
            
        if maze_copy[x][y] != 'S' and maze_copy[x][y] != 'E':
            maze_copy[x][y] = '*'
        
        if maze_copy[x][y] == 'E':
            break
    
    for row in maze_copy:
        print(' '.join(row))

print("\nBest path:")
visualize_path(solution)