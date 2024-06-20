from  utils import saveData, graficar, generateObjectivo, pintar
from PGenetica import PGenetica
from mpi4py import MPI
import numpy as np
import random

import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sizePoblacion = 100
limiteGeneraciones = 10

X = None
y = None
objGenetica = None
X, y = generateObjectivo()
operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos", "tan","log"]
objGenetica = PGenetica(X, y,operators,functions )

if rank == 0:
    # logging.debug('Iniciamos generando la funcion objectivo')
    poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
    saveData(poblacion,"Inicial")
else:
    poblacion = None

if rank == 0:
    sub_poblaciones = np.array_split(poblacion, size)
else:
    sub_poblaciones = None

#Enviamos y recibimos la informacion
sub_poblacion = comm.scatter(sub_poblaciones, root=0)
    
for i in range(limiteGeneraciones):
    nuevaGeneracion = objGenetica.generateGeneration(sub_poblacion)
    nuevaGeneracion = comm.gather(nuevaGeneracion, root=0)
    if rank == 0:
        # print("recibimos en el nodo 0 las genereaciones generadas")
        unimosPoblacion = [ind for sublist in nuevaGeneracion for ind in sublist]
        nuevaGeneracion = sorted(unimosPoblacion, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))
        #Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación
        poblacion = poblacion[:len(poblacion)//2] + nuevaGeneracion[:len(nuevaGeneracion)//2]
        sub_poblaciones = np.array_split(poblacion, size)
        numero_entero = random.randint(1, 100)
    else:
        # print(len(sub_poblacion))
        pass

    sub_poblacion = comm.scatter(sub_poblaciones, root=0)

poblacion_completa = comm.gather(sub_poblacion, root=0)
if rank == 0:
    # Concatenar todas las sublistas de poblacion_completa en una sola lista
    poblacionFinal = [ind for sublist in poblacion_completa for ind in sublist]
    # saveData(poblacionFinal,"Final")    
    y_ = [poblacionFinal[i]['y_predict'] for i in range(len(poblacion[:10]))]
    #numero random
    num = np.random.randint(1000)
    pintar(X,y,y_,f"Final_{num}")
    

