import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import random
import math
import logging
from colorlog import ColoredFormatter
np.seterr(divide='ignore', invalid='ignore')


def saveData(poblacion,name):
    df = pd.DataFrame(poblacion)
    timeNow = datetime.datetime.now()
    numero_aleatorio = random.randint(0, 100)
    # now = timeNow.strftime('%H_%M_%SS')
    df.to_csv(f"poblaciones/Poblacion_{name}.csv")

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
    plt.plot(X,y,linestyle='--', label='Funcion objectivo', color='r')
    for i, (y, color) in enumerate(zip(y_, colors)):
        plt.plot(X, y, linestyle='-', color=color)
    plt.title('Gráfico de varias listas en el eje y con colores aleatorios')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graficas/{name}.png', format='png')

def funciones():
    regressionFunctions = {
        "f1":{"fx":"X**3+X**2+X","fitCases":np.linspace(-1, 1, 5)},
        "f2":{"fx":"X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f3":{"fx":"X**5+X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f4":{"fx":"X**6+X**5+X**4+X**3+X**2+X","fitCases":np.linspace(-1, 1, 20)},
        "f5":{"fx":"sin(X**2)*cos(X)-1","fitCases":np.linspace(-1, 1, 20)},
        "f6":{"fx":"sin(X)+sin(X+X**2)","fitCases":np.linspace(-1, 1, 20)},
        "f7":{"fx":"log(X+1)+log(X**2 +1)","fitCases":np.linspace(0, 2, 20)},
    }
    return regressionFunctions

def evaluar_expresion(expresion,X):
    try:
        return round(eval(expresion, {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,"X": X , 'log' :math.log}),4 )
    except ZeroDivisionError:
        return 0
    except Exception as e:
        return 0

def normalize_list(lista):
    valores_validos = [valor for valor in lista if valor is not None]
    minimo = min(lista)
    maximo = max(lista)
    return [(x - minimo) / (maximo - minimo) if (maximo - minimo) != 0 else 0 for x in lista]   
   

def generateObjectivo():
    fxs = funciones()
    # i = f"f{np.random.randint(1,6)}"
    i = f"f{1}"
    X = fxs[i]["fitCases"]
    y  = [ (evaluar_expresion(fxs[i]["fx"], x)) for x in X ]
  
    return X, y, fxs[i]["fx"]

def generateLog():
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
    return logger
