from mpi4py import MPI
import numpy as np
import argparse
#Obtenemos por paramentro el nodo maestro
parser = argparse.ArgumentParser(description="Nodo Maestro")
parser.add_argument("-nodo", type=int, help="Nodo Maestro")
args = parser.parse_args()
masterNodo = args.nodo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm. Get_size()

t = 1
N = 10
n = 3

matrizNodos= np.arange(0, n*n).reshape(n,n)
coordenadas = np.where(matrizNodos == masterNodo)
coordenadas = list(zip(coordenadas[0], coordenadas[1]))
fila = matrizNodos[coordenadas[0][0],:]
columna = matrizNodos[:,coordenadas[0][1]]
new_size = len(fila)-1 + len(columna)-1

n_local = N // new_size
neighborhood = np.union1d(fila, columna)
neighborhood = np.delete(neighborhood, np.where(neighborhood == masterNodo))

poblacion_local = None
if rank in neighborhood:
    print("Entramos a los vecionos")
    poblacion_local = np.random.uniform(-10,10,n_local)

poblacion = None
if rank == masterNodo:
    poblacion = np.empty(n_local, dtype=np.float64)

comm.Gather(poblacion_local, poblacion, root=0)
if rank == 0:
    print(f"Proceso {rank}: Recolecté la población completa {poblacion}")