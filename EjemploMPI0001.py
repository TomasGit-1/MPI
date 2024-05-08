from mpi4py import MPI

comm = MPI.COMM_WORLD
#Numero de procesoso en los que se esta ejecutando
rank = comm.Get_rank()
#Total de procesos
size = comm.Get_size()

num = 12

if rank == 0:
    print("Estamos en el nodo maestro")
else:
    for i in range(rank, num, (size-1)):
        print(f"{i}", end=",")
    print("\n")


