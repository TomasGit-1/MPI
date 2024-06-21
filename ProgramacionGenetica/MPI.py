from  utils import saveData, graficar, generateObjectivo, pintar,normalize_list
from PGenetica import PGenetica
import numpy as np
import random
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sizePoblacion = 100
limiteGeneraciones = 10
value = np.random.randint(0, 200)
X = None
y = None
fx = None
operators = ["+", "-", "*", "/"]#**
functions = ["sin", "cos","log"] #,"log"

if rank == 0:
    value = np.random.randint(0, 200)
    X, y, fxs= generateObjectivo()

#EN todas los nodos 
X = comm.bcast(X, root=0)
y = comm.bcast(y, root=0)
objGenetica = PGenetica(X, y,operators,functions )
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)

for i in range(limiteGeneraciones):
    antiguaGeneracion = poblacion
    nuevaGeneracion =objGenetica.generateGeneration(antiguaGeneracion)

    if isinstance(antiguaGeneracion, np.ndarray):
        antiguaGeneracion = antiguaGeneracion.tolist()

    nuevaGeneracion = antiguaGeneracion[len(antiguaGeneracion)//2:] + nuevaGeneracion[len(nuevaGeneracion)//2:]
    nuevaGeneracion = np.array_split(nuevaGeneracion, size)
    nuevaGeneracionRecv = comm.gather(nuevaGeneracion[0], root=0)

    if rank == 0:
        poblacion_completa = [ind for sublist in nuevaGeneracionRecv for ind in sublist]
        poblacion_completa = sorted(poblacion_completa, key=lambda x: np.inf if x['mse'] is None or np.isnan(x['mse']) else x['mse'])
        # print(len(poblacion_completa))
        # poblacion = np.array_split(poblacion_completa, size)
        # poblacion = poblacion_completa[:sizePoblacion]
    else:
        poblacion_completa = None
    # poblacion = comm.scatter(poblacion_completa, root=0)
    poblacion = comm.bcast(poblacion, root=0)
    """
        Tomamso los mejores 50% de la Antigua Generacion y los 50% mejores de la nmueva generaion
    """
poblacion_completa = comm.gather(poblacion, root=0)

if rank == 0:
    print(value)
    graficar(X,y,f"{i}_{value}_{rank}",fxs)
    poblacionFinal = [ind for sublist in poblacion_completa for ind in sublist]
    # saveData(poblacionFinal,"Final")    
    y_predict = [poblacionFinal[i]['y_predict'] for i in range(len(poblacionFinal[:10]))]
    pintar(X, y, y_predict, f"Final_{value}")


