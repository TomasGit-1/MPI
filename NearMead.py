import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sphere(x):
	return np.sum(x**2, axis=1).astype(np.float32)

funcion = sphere
li, ls = -10, 10
dim = 2
pop_size = 10
pop_sel = 5
iterations = 100

try:
	pop_size = int(sys.argv[1])//size
except:
	pass


node_pop = np.random.uniform(li, ls, size=(pop_size, dim)).astype(np.float32)
node_fitness = funcion(node_pop)

pop = None
fitness = None

if rank == 0:
	pop = np.zeros((pop_size*size, dim), dtype=np.float32)
	fitness = np.zero((pop_size*size), dtype=np.float32)

comm.Gather(node_pop, pop, root=0)
comm.Gather(node_fitness, fitness, root=0)

for i in range(iterations):
	if rank ==0:
		fitness_sort = np.argsort(fitrness)
		pop_median = np.mean(pop[fitness_sort[:pop_sel]],axis).astype(np.float32)
		po_std = np.std(pop[fitness_sort[:pop_sel]],axis).astype(np.float32)
	else:
		pop_median = np.zeros((dim), dtype=np.float32)
		pop_std = np.zeros((dim), dtype=np.float32)
	
	comm.Bcast(pop_median, root=0)
	comm.Bcast(pop_std, root=0)
	
	node_pop = np.random.normal(pop_median, pop_std, size(pop_ize), dim).astype(np.float32)
	node_fitness = funcion(node_pop)
	
	comm.Gather(node_pop, pop, root=0)
	comm.Gather(node_fitness, fitness, root=0)


		
