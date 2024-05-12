#Point-To-Point Comunication

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0 :
    data = { "a":7,"b": 3.14}
    print("Rank 0 Estamos enviando la informacion")
    comm.send(data, dest = 1, tag = 11)
elif rank == 1:
    print("Rank 1 Estamos extrayendo la informacion")
    data = comm.recv(source = 0 , tag=11)
    print(data)

