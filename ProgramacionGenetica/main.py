from  utils import saveData, graficar, generateObjectivo, pintar,normalize_list
from PGenetica import PGenetica
import numpy as np
import random
import pandas as pd
sizePoblacion = 100
limiteGeneraciones = 10

value = np.random.randint(0, 200)
print(f"#_{value}")
X, y = generateObjectivo(value)
operators = ["+", "-", "*", "/"]#**
functions = ["sin", "cos","log"] #,"log"

objGenetica = PGenetica(X, y,operators,functions )
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)
saveData(poblacion,f"Inicial_{value}")

for i in range(limiteGeneraciones):
    print("Numero de generacion: " + str(i))
    antiguaGeneracion = poblacion
    NuevaGeneracion =objGenetica.generateGeneration(antiguaGeneracion)
    """
        Tomamso los mejores 50% de la Antigua Generacion y los 50% mejores de la nmueva generaion
    """
    poblacion = antiguaGeneracion[len(antiguaGeneracion)//2:] + NuevaGeneracion[len(NuevaGeneracion)//2:]
    # # poblacion = sorted(poblacion, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))
    poblacion = sorted(poblacion, key=lambda x: np.inf if x['mse'] is None or np.isnan(x['mse']) else x['mse'])

df = pd.DataFrame(poblacion)
y_predict = [poblacion[i]['y_predict'] for i in range(len(poblacion[:5]))]
# y_predict = [normalize_list(lista) for lista in y_predict]
saveData(poblacion,f"Final_{value}") 
pintar(X,y,y_predict,f"Final_{value}")