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
#print(f"Rank :{rank}")
#print(f"Rank :{size}")


n = 3
matrizNodos= np.arange(0, n*n).reshape(n,n)
print(matrizNodos)

#Encontramos las Coordenadas del Nodo Maestro

coordenadas = np.where(matrizNodos == masterNodo)
coordenadas = list(zip(coordenadas[0], coordenadas[1]))

fila = matrizNodos[coordenadas[0][0],:]
print(fila)
columna = matrizNodos[:,coordenadas[0][1]]
print(columna)
#Media y desviacion estandar  

#Enviarlos a los nodos vecinos para que ellos generen una nueva poblacion 

#Cada Nodo debe de tener un aleaotira 100

