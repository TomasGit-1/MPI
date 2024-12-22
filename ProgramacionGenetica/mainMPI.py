#Class and util
from  utils import saveData, graficar, generateObjectivo, pintar,generateLog
from PGenetica import PGenetica

#Librerias
from mpi4py import MPI
import numpy as np
import random
import logging
from colorlog import ColoredFormatter

log = generateLog()
start_time = MPI.Wtime()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


sizePoblacion = 400
limiteGeneraciones = 10
X = None
y = None
objGenetica = None
operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos", "tan","log"]
value = 0
limite = 0.01
funcioN = 7
if rank == 0:
    log.info("En el nodo 0 Generamos la funcion objectivo")
    value = np.random.randint(0, 10000)
    X, y, fxs, nf= generateObjectivo(num = funcioN)

X = comm.bcast(X, root=0)
y = comm.bcast(y, root=0)
objGenetica = PGenetica(log,X, y,operators,functions)

if rank == 0:
    log.info("generando la poblacion")
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

nuevaGeneracion = poblacion
mse = 10
i = 0 
cercano =None

log.warning("Generando nuevao poblacion")
while mse > limite:
    nuevaGeneracion = objGenetica.generateGeneration(sub_poblacion)
    nuevaGeneracion = comm.gather(nuevaGeneracion, root=0)
    if rank == 0:
        log.warning("Recibimos en el nodo 0 las genereaciones generadas")
        # print("recibimos en el nodo 0 las genereaciones generadas")
        nuevaGeneracion = [ind for sublist in nuevaGeneracion for ind in sublist]
        nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
        log.warning("Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación")
        poblacion = poblacion[:len(poblacion)//2] + nuevaGeneracion[:len(nuevaGeneracion)//2]
        log.warning("Volvemos a ordenar")
        nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
        cercano = nuevaGeneracion[0]
        mse = cercano["mse"]
        log.warning(f"MSE generacion {i}: {mse}")
        i += 1
        sub_poblaciones = np.array_split(poblacion, size)
        # sub_poblaciones = poblacion
        numero_entero = random.randint(1, 100)
    else:
        pass
    mse = comm.bcast(mse, root=0)
    sub_poblacion = comm.scatter(sub_poblaciones, root=0)

# poblacion_completa = comm.gather(sub_poblacion, root=0)

if rank == 0:
    # Concatenar todas las sublistas de poblacion_completa en una sola lista
    # poblacionFinal = [ind for sublist in poblacion_completa for ind in sublist]
    log.info(f"Imagenen generada en {value}")
    y_ = cercano['y_predict']
    expresion = cercano['expresion']
    objectivo = fxs
    saveData(cercano,f"Final_{value}_{nf}") 
    pintar(X,y,y_,f"Final_{value}_{nf}", objectivo, expresion)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    log.info(f"Tiempo transcurrido: {elapsed_time:.6f} segundos")
    
        


