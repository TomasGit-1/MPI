from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10
if rank == 0:
    print("Entramos al Nodo Maestro")
    data = np.random.uniform(0,10,N)
    # print(f"Data Generada: {data}")
else:
    data = None

dataDividido = np.empty(N//size, dtype = float)
comm.Scatter(data, dataDividido, root=0 )
suma = np.sum(dataDividido)
# print(f"Rank {rank} data {suma}")

sumaTotal = comm.gather(suma, root = 0)
if rank == 0:
    print(f"Suma Total {sumaTotal}")
    print(f"Suma Total {np.sum(sumaTotal)}")