from  utils import saveData, graficar, generateObjectivo, pintar
from PGenetica import PGenetica
import numpy as np
import random
import pandas as pd
sizePoblacion = 100
limiteGeneraciones = 10

X, y = generateObjectivo()

operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos"] #,"log"

objGenetica = PGenetica(X, y,operators,functions )
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
saveData(poblacion,"Inicial")

for i in range(limiteGeneraciones):
    antiguaGeneracion = poblacion
    NuevaGeneracion =objGenetica.generateGeneration(antiguaGeneracion)
    """
        Tomamso los mejores 50% de la Antigua Generacion y los 50% mejores de la nmueva generaion
    """
    poblacion = antiguaGeneracion[len(antiguaGeneracion)//2:] + NuevaGeneracion[len(NuevaGeneracion)//2:]
    print(len(poblacion))
    poblacion = sorted(poblacion, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))

df = pd.DataFrame(poblacion)
y_predict = [poblacion[i]['y_predict'] for i in range(len(poblacion[:5]))]

saveData(poblacion,"Final") 
pintar(X,y,y_predict,f"Final_{1}")