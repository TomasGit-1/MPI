import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import random
import math
np.seterr(divide='ignore', invalid='ignore')


def saveData(poblacion,name):
    df = pd.DataFrame(poblacion)
    timeNow = datetime.datetime.now()
    numero_aleatorio = random.randint(0, 100)
    # now = timeNow.strftime('%H_%M_%SS')
    df.to_csv(f"poblaciones/Poblacion_{name}_{numero_aleatorio}.csv")

def graficar(X,y,name, expresion):
    plt.figure(figsize=(10, 6))
    plt.plot(X,y, label='Funcion', color='r')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Gráfico de y = {expresion}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graficas/{name}.png', format='png')

def pintar(X,y,y_,name):
    colors = np.random.rand(len(y_), 3)
    plt.figure(figsize=(10, 6))
    plt.plot(X,y, label='Funcion objectivo', color='r')
    for i, (y, color) in enumerate(zip(y_, colors)):
        print(i)
        plt.plot(X, y, linestyle='-', color=color)
    plt.title('Gráfico de varias listas en el eje y con colores aleatorios')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graficas/{name}.png', format='png')

def funciones():
    regressionFunctions = {
        "f1":{"fx":"X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f2":{"fx":"X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f3":{"fx":"X**5+X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f4":{"fx":"X**6+X**5+X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f5":{"fx":"sin(X**2)*cos(X)-1","fitCases":np.linspace(-1, 1, 20)},
        "f6":{"fx":"sin(X)+sin(X+X**2)","fitCases":np.linspace(-1, 1, 20)},
        "f7":{"fx":"log(X+1)+log(X**2 +1)","fitCases":np.linspace(0, 2, 20)},
    }
    return  regressionFunctions

def evaluar_expresion(expresion,X):
        try:
            return round(eval(expresion, {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,"X": X , 'log' :math.log}),4 )
        except ZeroDivisionError:
            return None
        except Exception as e:
            return None
        
def generateObjectivo():
    np.random.seed(42)
    fxs = funciones()
    # for i in fxs.keys():
    i = f"f{np.random.randint(1,6)}"
    X = fxs["f5"]["fitCases"]
    y  = [ (evaluar_expresion(fxs[i]["fx"], x)) for x in X ]
    #Normalizamos y entre 0 y 1
    y = [(valor -  min(y)) / ( max(y) - min(y)) for valor in y]

    graficar(X,y,i,fxs[i]["fx"])
    return X, y, 

