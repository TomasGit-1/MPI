from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
	data = "Mensaje desde el proceso 0"
else:
	#En los otros procesos no se genera nada
	data = None

data = comm.bcast(data, root=0)

print(f"Proceso {rank} ha recibido el mensaje: {data}")
