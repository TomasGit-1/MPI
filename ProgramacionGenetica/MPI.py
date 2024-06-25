from  utils import saveData, graficar, generateObjectivo, pintar,normalize_list
from PGenetica import PGenetica
import numpy as np
import random
import pandas as pd
from mpi4py import MPI
import logging
from colorlog import ColoredFormatter

logger = logging.getLogger()
logger.setLevel(logging.INFO) 

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
log = logger
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sizePoblacion = 100
limiteGeneraciones = 20
value = np.random.randint(0, 200)
X = None
y = None
fx = None
operators = ["+", "-", "*", "/"]#**
functions = ["sin", "cos","log"] #,"log"

if rank == 0:
    log.info("En el nodo 0 Generamos la funcion objectivo")
    value = np.random.randint(0, 200)
    X, y, fxs= generateObjectivo()

#EN todas los nodos 
log.warning("En todos los nodos generamos la poblacion alaeatoria")
X = comm.bcast(X, root=0)
y = comm.bcast(y, root=0)

objGenetica = PGenetica(X, y,operators,functions )
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
log.warning(f"Generamos poblacion en {rank} {len(poblacion)}")

for i in range(limiteGeneraciones):
    antiguaGeneracion = poblacion
    nuevaGeneracion =objGenetica.generateGeneration(antiguaGeneracion)

    if isinstance(antiguaGeneracion, np.ndarray):
        antiguaGeneracion = antiguaGeneracion.tolist()
    """
        Tomamso los mejores 50% de la Antigua Generacion y los 50% mejores de la nmueva generaion
    """
    nuevaGeneracion = antiguaGeneracion[len(antiguaGeneracion)//2:] + nuevaGeneracion[len(nuevaGeneracion)//2:]
    # nuevaGeneracion = np.array_split(nuevaGeneracion, size)
    nuevaGeneracionRecv = comm.gather(nuevaGeneracion, root=0)

    if rank == 0:
        poblacion_completa = [ind for sublist in nuevaGeneracionRecv for ind in sublist]
        poblacion_completa = sorted(poblacion_completa, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))
        poblacion = np.array_split(poblacion_completa, size)
    else:
        pass
    poblacion = comm.scatter(poblacion, root=0)
  
poblacion_completa = comm.gather(poblacion, root=0)

if rank == 0:
    print(value)
    poblacionFinal = [ind for sublist in poblacion_completa for ind in sublist]
    # saveData(poblacionFinal,"Final1")    
    y_predict = [poblacionFinal[i]['y_predict'] for i in range(len(poblacionFinal[:10]))]
    graficar(X,y,f"{i}_{value}_{rank}",fxs)
    pintar(X, y, y_predict, f"Final_{value}")


