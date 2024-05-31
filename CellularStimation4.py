from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def estimate_gaussian(selected_individuals):
    mean = np.mean(selected_individuals)
    std = np.std(selected_individuals)
    return mean, std

def sphere(x):
	return np.sum(x**2, axis=1).astype(np.float32)

pop_size = 10
try:
	pop_size = int(sys.argv[1])//size
except:
	pass

funcion = sphere

N = 5
li = -10
ls = 10
# poblacion = np.random.uniform(li, ls, size=(pop_size, 2)).astype(np.float32)
# poblacion_fitness = funcion(poblacion)
# print(f"Nodo {poblacion}")
# print(f"Fitness {poblacion_fitness}")

if rank == 0:
    poblacion = np.random.rand(N)  # Población inicial aleatoria
    # poblacion = np.random.uniform(li, ls, size=(pop_size, 2)).astype(np.float32)
    # poblacion = np.random.uniform(li, ls,N).astype(np.float32)
else:
    poblacion = None

#bcast transmite la misma poblacion a todo
poblacion = comm.bcast(poblacion, root=0)
# poblacion = comm.bcast(poblacion_fitness, root=0)
poblacion_local = poblacion
max_iteraciones = 1
tam_cell = 5
neighborhood_size = 2
t = 0

while t < max_iteraciones:
    new_population_local = []
    #Esto lo hara en cada uno de los Nodos
    for i in range(0, len(poblacion_local), tam_cell):
        fitness_sort = np.argsort(poblacion_local)
        cell = poblacion_local[i:i+tam_cell]
        M = min(neighborhood_size * tam_cell, len(cell))
        selected_individuals = np.random.choice(cell, M)
        mean, std = estimate_gaussian(selected_individuals)
        new_individuals = np.random.normal(mean, std, len(cell))
        new_population_local.extend(new_individuals)
    # print(f"Rank {rank}")
    # print(new_population_local)

    #Recopilamos en el nodo 0
    new_population = comm.gather(new_population_local, root=0)

    if rank == 0:
        print(3*"-------")
        # node_fitness = funcion(new_population[0])
        # new_population = np.concatenate(new_population)
        # node_fitness = funcion(node_pop)
        print(new_population[0])

    else:
        new_population = None
    poblacion_local = comm.scatter(new_population[0], root=0)
    print(poblacion_local)

    t += 1