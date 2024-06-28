#Class and util
from  utils import saveData, graficar, generateObjectivo, pintar,generateLog
from PGenetica import PGenetica

#Librerias
import numpy as np
import random
log = generateLog()

sizePoblacion = 100
limiteGeneraciones = 100
X = None
y = None
objGenetica = None
operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos", "tan","log"]
value = 0

log.info("En el nodo 0 Generamos la funcion objectivo")
value = np.random.randint(0, 200)
X, y, fxs= generateObjectivo()

objGenetica = PGenetica(log, X, y,operators,functions)

log.info("generando la poblacion")
value = np.random.randint(0, 200)
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)

log.info("Generamos la sub poblaciones")

#Enviamos a los nodos las subPoblaciones y recibimos la informacion
log.warning("Enviamos a los nodos las subPoblaciones y recibimos la informacion")

log.warning("Generando nuevao poblacion")

nuevaGeneracion = poblacion
for i in range(limiteGeneraciones):
    nuevaGeneracion = objGenetica.generateGeneration(nuevaGeneracion)
    log.warning("Recibimos en el nodo 0 las genereaciones generadas")
    # print("recibimos en el nodo 0 las genereaciones generadas")
    # unimosPoblacion = [ind for sublist in nuevaGeneracion for ind in sublist]
    nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
    log.warning("Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación")
    nuevaGeneracion = poblacion[:len(poblacion)//2] + nuevaGeneracion[:len(nuevaGeneracion)//2]
    log.warning("Volvemos a ordenar")
    nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
    y_ = [nuevaGeneracion[i]['y_predict'] for i in range(len(nuevaGeneracion[:2]))]

    log.warning(f"MSE : {y_}")
    # sub_poblaciones = poblacion
    # numero_entero = random.randint(1, 100)


log.info(f"Imagenen generada en {value}")
# Concatenar todas las sublistas de poblacion_completa en una sola lista
y_ = [nuevaGeneracion[i]['y_predict'] for i in range(len(nuevaGeneracion[:1]))]
# log.info(f"Mejores {y_}")
graficar(X,y,f"{i}_{value}",fxs)
pintar(X,y,y_,f"Final_{value}")
    

