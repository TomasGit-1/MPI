from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	data = list(range(size))
	print(data)
else:
	data = None

data = comm.scatter(data, root=0)

print(f"Proceso {rank} recibio el dato: {data}")

data *= 2

getData = comm.gather(data, root=0)

if rank == 0:
	print(f"Datos recogidos: {getData}")
