from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm. Get_size()
#print(f"Rank :{rank}")
#print(f"Rank :{size}")


print("Estamos en el nodo Maestro")
n = 3
poblacion = np.arange(0, n*n).reshape(n,n)
print(poblacion)


#Media y desviacion estandar  

#Enviarlos a los nodos vecinos para que ellos generen una nueva poblacion 

#Cada Nodo debe de tener un aleaotira 100

