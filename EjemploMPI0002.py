from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Ejmplo de uso de Send()  y Recv()

print("Ejmplo de Send y Recv")
if rank == 0 :
    data = {"message": "Hola desde el proceso 0"}
    comm.Send(data, dest = 1, tag = 11)
elif rank == 1:
    data = [None]
    comm.Recv(data, source = 0, tag = 11)
    #print(f"Proceso {rank} ha recibido el mensaje : {data['message']}")


