from PGenetica import PGenetica
import numpy as np
from mpi4py import MPI
import random
import pandas as pd
comm = MPI.COMM_WORLD
import datetime
rank = comm.Get_rank()
size = comm.Get_size()

def saveData(poblacion,name):
    df = pd.DataFrame(poblacion)
    timeNow = datetime.datetime.now()
    numero_aleatorio = random.randint(0, 100)
    # now = timeNow.strftime('%H_%M_%SS')
    df.to_csv(f"poblaciones/Poblacion_{name}_{numero_aleatorio}.csv")

# Generar datos de entrada (X)
np.random.seed(42)
X = np.linspace(1, 2 * np.pi, 100)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)
sizePoblacion = 200
limiteGeneraciones = 10
objGenetica = PGenetica(X, y, limite=200)

if rank == 0:
    poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
    saveData(poblacion,"Inicial")
else:
    poblacion = None

if rank == 0:
    sub_poblaciones = np.array_split(poblacion, size)
else:
    sub_poblaciones = None

#Recinimos la informacion
sub_poblacion = comm.scatter(sub_poblaciones, root=0)
    
for i in range(limiteGeneraciones):
    nuevaGeneracion = objGenetica.generateGeneration(sub_poblacion)
    nuevaGeneracion = comm.gather(nuevaGeneracion, root=0)
    if rank == 0:
        print("recibimos en el nodo 0 las genereaciones generadas")
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
    saveData(poblacionFinal,"Final")
    print(len(poblacionFinal))
    

