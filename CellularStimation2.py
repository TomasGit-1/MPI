import numpy as np

def estimate_gaussian(selected_individuals):
    mean = np.mean(selected_individuals)
    std = np.std(selected_individuals)
    return mean, std

t = 1
N = 10
max_iteraciones = 10
tam_cell = 2
neighborhood_size = 10
poblacion = np.random.uniform(-10,10,N)

while t <  max_iteraciones:
    new_population = []
    for i in range(0, N,tam_cell):
        cell = poblacion[i:i+tam_cell]
        M = min(neighborhood_size * tam_cell, len(cell))
        selected_individuals = np.random.choice(cell, M)
        mean, std = estimate_gaussian(selected_individuals)
        new_individuals = np.random.normal(mean, std, len(cell))
        new_population.extend(new_individuals)
    poblacion = np.array(new_population)
    t += 1