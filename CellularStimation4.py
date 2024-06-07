import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Set_errhandler(MPI.ERRORS_RETURN)

def getNeighborhood(masterNodo,sizeM):
    sizeM = int((sizeM/2))
    matrizNodos= np.arange(0, sizeM**2).reshape(sizeM,sizeM)
    # print(matrizNodos)
    coordenadas = np.where(matrizNodos == masterNodo)
    coordenadas = list(zip(coordenadas[0], coordenadas[1]))
    fila = matrizNodos[coordenadas[0][0],:]
    columna = matrizNodos[:,coordenadas[0][1]]
    neighborhood = np.union1d(fila, columna)
    neighborhood = np.delete(neighborhood, np.where(neighborhood == masterNodo))
    return {masterNodo:neighborhood}

def sphere(x):
    return np.sum(x**2, axis=1).astype(np.float32)

def elegirIndicesMejores(M, node_fitness): 
    #Obtenemos los indices de M
    return np.argsort(node_fitness)[-M:]

funcion = sphere
li, ls = -10, 10
dim = 2
pop_size = 5
max_iteraciones = 2
t = 1
SizeOf_cell = 2
neighborhood = getNeighborhood(rank,size)
SizeOf_Neighborhood  = neighborhood[rank].shape[0]

# print(f"Estamos en rankclea {rank}")
# print(f"Estos son los vecinos {neighborhood} Size {neighborhood_size}")

#Genereamos la poblacion aleatoria para cada nodo y Evaluamos la funcion objectivo
node_pop = np.random.uniform(li, ls, size=(pop_size, dim)).astype(np.float32)
node_fitness = funcion(node_pop)

pop = None
fitness = None
# print(neighborhood.keys())
if rank==0:
    pop = np.zeros((pop_size*size, dim), dtype=np.float32)
    fitness = np.zeros((pop_size*size), dtype=np.float32)


while t <  max_iteraciones:
    #Enviamos de mi rank actual , los mejores elementosa sus vecionos
    """
        Aqui tenemos que elegir los M Mejores
    """
    mejores = elegirIndicesMejores(2,node_fitness)
    for target in neighborhood[rank]:
        print(f"Estamos en rank {rank} Con vecinos {neighborhood[rank]}  Enviando a ... {target}" )
        comm.Send([node_pop[mejores], MPI.FLOAT], dest=target)
        comm.Send([node_fitness[[mejores]], MPI.FLOAT], dest=target)


    node_pop_local = np.random.uniform(li, ls, size=(pop_size, dim)).astype(np.float32)
    node_fitness_local = funcion(node_pop_local)
    
    for source in neighborhood[rank]:
        print(f"Estamos en rank {rank} Con vecinos {neighborhood[rank]}  Recibiendo en  ... {source}" )
        #Generamos la poblacions local

        #TEnsmo que crear un array para recibir los datos
        node_pop_visitante = np.empty((pop_size, dim), dtype=np.float32)
        node_fitness_visitante = np.empty(pop_size, dtype=np.float32) 

        comm.Recv([node_pop_visitante, MPI.FLOAT], source=source)
        comm.Recv([node_fitness_visitante, MPI.FLOAT], source=source)
        node_fitness_visitante = funcion(node_pop_visitante)

    #Mensaliamo los vecionos recibidos u los locales 
    node_full = np.concatenate((node_pop_local, node_pop_visitante), axis=0)
    node_fitness_full = np.concatenate((node_fitness_local, node_fitness_visitante), axis=0)
    #Seleccionamos los mejores 
    mejores = elegirIndicesMejores(2,node_fitness)

    pop_median = np.mean(node_full[mejores], axis=0).astype(np.float32)
    pop_std = np.std(node_fitness_full[mejores], axis=0).astype(np.float32)
    
    node_pop = np.random.normal(pop_median, pop_std, size=(pop_size, dim)).astype(np.float32)
    node_fitness = funcion(node_pop)

    comm.Send([node_pop, MPI.FLOAT], dest=0, tag=rank)
    comm.Send([node_fitness, MPI.FLOAT], dest=0, tag=rank)


    comm.Gather(node_pop, pop, root=0)
    comm.Gather(node_fitness, fitness, root=0)

    if rank == 0:
        print("En el nodo 0 Obtenemos todos")
        print(pop)
        print(fitness)
        # mejores = elegirIndicesMejores(2,fitness)
        # print(f"Mejores Datos {node_full[mejores]} ")

    t +=1

