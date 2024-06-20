import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import random


def saveData(poblacion,name):
    df = pd.DataFrame(poblacion)
    timeNow = datetime.datetime.now()
    numero_aleatorio = random.randint(0, 100)
    # now = timeNow.strftime('%H_%M_%SS')
    df.to_csv(f"poblaciones/Poblacion_{name}_{numero_aleatorio}.csv")

def graficar(X,y,name):
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    # plt.plot(X, y, label='Datos con ruido', color='b', marker='o', linestyle='none')
    plt.plot(X,y, label='Seno', color='r')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gráfico de y = sin(X)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graficas/{name}.png', format='png')

def pintar(X,y,y_,name):
    colors = np.random.rand(len(y_), 3)
    plt.figure(figsize=(10, 6))
    plt.plot(X,y, label='Funcion objectivo', color='r')
    for i, (y, color) in enumerate(zip(y_, colors)):
        # label = f'Datos {i + 1}'
        plt.plot(X, y, linestyle='--', color=color)

    plt.title('Gráfico de varias listas en el eje y con colores aleatorios')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graficas/{name}.png', format='png')


# Mostrar la gráfica
plt.show()
def f(x):
    # return np.sin(x)
    return x**3 + x**2 + x
    # return np.log(np.abs(x) + 1)
    
def generateObjectivo():
    np.random.seed(42)
    X = np.linspace(-1, 1, 20)
    y = f(X)
    print(len(y))
    return X, y

