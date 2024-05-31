from mpi4py import MPI
import numpy as np
import argparse
#Obtenemos por paramentro el nodo maestro
parser = argparse.ArgumentParser(description="Nodo Maestro")
parser.add_argument("-nodo", type=int, help="Nodo Maestro")
args = parser.parse_args()
masterNodo = args.nodo

def getNeighborhood(masterNodo,sizeM):
    matrizNodos= np.arange(0, sizeM*sizeM).reshape(sizeM,sizeM)
    coordenadas = np.where(matrizNodos == masterNodo)
    coordenadas = list(zip(coordenadas[0], coordenadas[1]))
    fila = matrizNodos[coordenadas[0][0],:]
    columna = matrizNodos[:,coordenadas[0][1]]
    return fila, columna

def estimate_gaussian(selected_individuals):
    mean = np.mean(selected_individuals)
    std = np.std(selected_individuals)
    return mean, std

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm. Get_size()

#N es el numero de muestras
N = 1000
#sizeM Es el tamanio de la Matriz de nodos
sizeM = 2

fila, columna = getNeighborhood(masterNodo,sizeM)
#A que nodo vamos a enviar
neighborhood = np.union1d(fila, columna)
neighborhood = np.delete(neighborhood, np.where(neighborhood == masterNodo))
population = np.random.uniform(-10,10,N)

stats = None
poblacion_local = None
poblacion = None
sub_population_size = N // len(neighborhood)

if rank == masterNodo:
    sub_populations = [population[i * sub_population_size: (i + 1) * sub_population_size] for i in range(len(neighborhood))]
    for i, node in enumerate(neighborhood):
        comm.send(sub_populations[i], dest=node, tag=77)

if rank in neighborhood:
    print(f"Entramos a los vecinos {rank}")
    population_local = np.random.uniform(-10,10,N)
    sub_population = comm.recv(source=masterNodo, tag=77)
    mean, std= estimate_gaussian(sub_population)
    stats = {"mean":mean, "std":std}

all_stats = comm.gather(stats, root=masterNodo)
if rank == masterNodo:
    print(all_stats)