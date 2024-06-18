from PGenetica import PGenetica
from binarytree import Node,build
import numpy as np
import pandas as pd

# Generar datos de entrada (X)
np.random.seed(42)
X = np.linspace(1, 2 * np.pi, 100)
# Añadir ruido a los datos3
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

objGenetica = PGenetica(X,y,limite=200)
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=100,profundidad=4)
print("Primera geneacion")

for i in range(10):
    antiguaGeneracion = poblacion
    NuevaGeneracion =objGenetica.generateGeneration(antiguaGeneracion)
    """
        Tomamso los mejores 50% de la Antigua Generacion y los 50% mejores de la nmueva generaion
    """
    poblacion = antiguaGeneracion[len(antiguaGeneracion)//2:] + NuevaGeneracion[len(NuevaGeneracion)//2:]

# Convertir la lista de diccionarios a DataFrame
df = pd.DataFrame(poblacion)
df.to_csv("poblacion.csv")
