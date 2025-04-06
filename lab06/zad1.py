import pandas as pd
import numpy
import pygad
import time

data = {
    "przedmiot": [
        "zegar", "obraz-pejzaż", "obraz-portret", "radio", "laptop", 
        "lampka nocna", "srebrne sztućce", "porcelana", "figura z brązu", 
        "skórzana torebka", "odkurzacz"
    ],
    "wartość": [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300],
    "waga": [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
}

df = pd.DataFrame(data, index=range(1, 12))
print(df)

def fitness_func(pyGad_class, solution, solution_idx):
    sum_value = numpy.sum(solution * df["wartość"])
    sum_weight = numpy.sum(solution * df["waga"])
    if sum_weight > 25:
        fitness = -1
    else:
        fitness = sum_value
    return fitness

gene_space = [0, 1]
sol_per_pop = 10
num_genes = len(data["przedmiot"])

print("Liczba genów: ", num_genes)

fitness_function = fitness_func
num_parents_mating = 5
num_generations = 100
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 8


target_fitness = 1630
successful_runs = 0
num_runs = 10
successful_times = []

for run in range(num_runs):
    start = time.time()
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
    stop_criteria=[f"reach_{target_fitness}"],)
    #uruchomienie algorytmu
    ga_instance.run()

    end = time.time()
    #podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness == target_fitness:
        successful_runs += 1
        successful_times.append(end - start)

successful_percentage = (successful_runs / num_runs) * 100
print("Percentage of successful runs: ", successful_percentage, "%")
print("Average time of successful runs: ", numpy.mean(successful_times), "seconds")

print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print("\nBest configuration of the knapsack problem:")
for i in range(len(solution)):
    if solution[i] == 1:
        print("Item: {item}, Value: {value}, Weight: {weight}".format(item=data["przedmiot"][i], value=data["wartość"][i], weight=data["waga"][i]))


print("\n")
#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(df["wartość"] *solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()