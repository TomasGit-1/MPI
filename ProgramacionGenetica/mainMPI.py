from PGenetica import PGenetica
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generar datos de entrada (X)
np.random.seed(42)
X = np.linspace(1, 2 * np.pi, 100)
# Añadir ruido a los datos3
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

objGenetica = PGenetica(X, y, limite=200)

print("Primera generacion")
if rank == 0:
    poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=100, profundidad=4)
else:
    poblacion = None

print("Dividir la población entre los nodos")
# Dividir la población entre los nodos
if rank == 0:
    sub_poblaciones = np.array_split(poblacion, size)
else:
    sub_poblaciones = None

sub_poblacion = comm.scatter(sub_poblaciones, root=0)

for i in range(10):
    antiguaGeneracion = sub_poblacion
    nuevaGeneracion = objGenetica.generateGeneration(antiguaGeneracion)
    
    NuevaG = comm.gather(nuevaGeneracion, root=0)
    if rank == 0:
        # Desempaquetar la lista anidada de NuevaG
        nuevaGeneracion_concatenada = [ind for sublist in NuevaG for ind in sublist]
        print(len(nuevaGeneracion_concatenada))
        print(len(antiguaGeneracion))
        # Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación
        sub_poblacion = poblacion[:len(poblacion)//2] + nuevaGeneracion_concatenada[:len(nuevaGeneracion_concatenada)//2]
    
    # Difundir la sub_poblacion actualizada
    sub_poblacion = comm.bcast(sub_poblacion, root=0)

# Recolectar la población completa en el nodo raíz
poblacion_completa = comm.gather(sub_poblacion, root=0)
if rank == 0:
    # Concatenar todas las sublistas de poblacion_completa en una sola lista
    poblacion_completa_flat = [ind for sublist in poblacion_completa for ind in sublist]
    
    # Convertir a DataFrame y escribir a CSV
    df = pd.DataFrame(poblacion_completa_flat)
    numero_entero = random.randint(1, 100)  # Genera un número entero aleatorio entre 1 y 100 (ambos inclusive)
    df.to_csv(f"poblacionCompleta{str(numero_entero)}.csv")
