#Class and util
from  utils import saveData, graficar, generateObjectivo, pintar,generateLog
from PGenetica import PGenetica

#Librerias
import numpy as np
import random
log = generateLog()

sizePoblacion = 400
limiteGeneraciones = 30
X = None
y = None
objGenetica = None
operators = ["+", "-", "*", "/","**"]
functions = ["sin", "cos", "tan","log"]
value = 0

log.info("En el nodo 0 Generamos la funcion objectivo")
value = np.random.randint(0, 200)
X, y, fxs, nf= generateObjectivo(num = 7)

objGenetica = PGenetica(log, X, y,operators,functions)

log.info("generando la poblacion")
value = np.random.randint(0, 200)
poblacion = objGenetica.generatePoblacionAleatoria(poblacionSize=sizePoblacion, profundidad=4)

log.info("Generamos la sub poblaciones")

#Enviamos a los nodos las subPoblaciones y recibimos la informacion
log.warning("Enviamos a los nodos las subPoblaciones y recibimos la informacion")

log.warning("Generando nuevao poblacion")

nuevaGeneracion = poblacion
limite = 0.01
mse = 10
i = 0 
cercano =None
while mse > limite:
    nuevaGeneracion = objGenetica.generateGeneration(nuevaGeneracion)
    log.warning("Recibimos en el nodo 0 las genereaciones generadas")
    # print("recibimos en el nodo 0 las genereaciones generadas")
    # unimosPoblacion = [ind for sublist in nuevaGeneracion for ind in sublist]
    nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
    log.warning("Tomar los mejores 50% de la Antigua Generación y los 50% mejores de la Nueva Generación")
    nuevaGeneracion = poblacion[:len(poblacion)//2] + nuevaGeneracion[:len(nuevaGeneracion)//2]
    log.warning("Volvemos a ordenar")
    nuevaGeneracion = objGenetica.ordenarPoblacion(nuevaGeneracion)
    cercano = nuevaGeneracion[0]
    mse = cercano["mse"]
    log.warning(f"MSE generacion {i}: {mse}")
    i += 1


log.info(f"Imagenen generada en {value}")
y_ = cercano['y_predict']
expresion = cercano['expresion']
objectivo = fxs
saveData(cercano,f"Final_{value}_{nf}") 
pintar(X,y,y_,f"Final_{value}_{nf}", objectivo, expresion)
    

