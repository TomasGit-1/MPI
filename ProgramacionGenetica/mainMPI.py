#Class and util
from  utils import saveData, graficar, generateObjectivo, pintar
from PGenetica import PGenetica

#Librerias
from mpi4py import MPI
import numpy as np
import random
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
limiteGeneraciones = 10
X = None
y = None
objGenetica = None
operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos", "tan","log"]
value = 0

if rank == 0:
    log.info("En el nodo 0 Generamos la funcion objectivo")
    value = np.random.randint(0, 200)
    X, y, fxs= generateObjectivo()

X = comm.bcast(X, root=0)
y = comm.bcast(y, root=0)
objGenetica = PGenetica(X, y,operators,functions)

if rank == 0:
    logger.info("generando la poblacion")
    value = np.random.randint(0, 200)
    poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
else:
    poblacion = None

if rank == 0:
    log.info("Generamos la sub poblaciones")
    sub_poblaciones = np.array_split(poblacion, size)
else:
    sub_poblaciones = None

#Enviamos a los nodos las subPoblaciones y recibimos la informacion
log.warning("Enviamos a los nodos las subPoblaciones y recibimos la informacion")
sub_poblacion = comm.scatter(sub_poblaciones, root=0)

log.warning("Generando nuevao poblacion")
for i in range(limiteGeneraciones):
    nuevaGeneracion = objGenetica.generateGeneration(sub_poblacion)
    nuevaGeneracion = comm.gather(nuevaGeneracion, root=0)
    if rank == 0:
        log.warning("Recibimos en el nodo 0 las genereaciones generadas")
        # print("recibimos en el nodo 0 las genereaciones generadas")
        unimosPoblacion = [ind for sublist in nuevaGeneracion for ind in sublist]
        nuevaGeneracion = sorted(unimosPoblacion, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))
        log.warning("Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación")
        poblacion = poblacion[:len(poblacion)//2] + nuevaGeneracion[:len(nuevaGeneracion)//2]
        sub_poblaciones = np.array_split(poblacion, size)
        # sub_poblaciones = poblacion
        numero_entero = random.randint(1, 100)
    else:
        pass

    sub_poblacion = comm.scatter(sub_poblaciones, root=0)

poblacion_completa = comm.gather(sub_poblacion, root=0)

if rank == 0:
    log.info(f"Imagenen generada en {value}")
    # Concatenar todas las sublistas de poblacion_completa en una sola lista
    poblacionFinal = [ind for sublist in poblacion_completa for ind in sublist]
    y_ = [poblacionFinal[i]['y_predict'] for i in range(len(poblacionFinal[:10]))]
    graficar(X,y,f"{i}_{value}",fxs)
    pintar(X,y,y_,f"Final_{value}")
    

