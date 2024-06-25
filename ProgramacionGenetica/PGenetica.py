from binarytree import Node,build
from itertools import combinations
from TreeC import TreeC
import numpy as np
import random
import copy
import math
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

class PGenetica:
    def __init__(self,X_true,y_true, operators=[],functions=[]):
        self.operators = operators
        self.variables = ["X"]
        self.functions = functions
        self.constants = [1]
        self.X_true = X_true
        self.y_true = y_true
        self.objTree = TreeC(self.operators, self.functions)

    def generatePoblacionAleatoria(self, poblacionSize = 4, profundidad=4):
        poblacion = []
        i=0
        while i < poblacionSize:
            Tree = self.objTree.build(profundidad)
            expresion,y_predict,mse,isValida = self.generateInfo(Tree)
            poblacion.append({"tree":Tree, "expresion":expresion,"y_predict":y_predict,"mse":mse, "isValida":isValida})
            i+=1
        poblacion = sorted(poblacion, key=lambda x: np.inf if x['mse'] is None or np.isnan(x['mse']) else x['mse'])
        # poblacion = sorted(poblacion, key=lambda x: x['mse'] )
        return poblacion
    
    def generateInfo(self, Tree):
        expresion,y_predict,mse,isValida = None,None,None, False
        try:
            expresion =  self.objTree.generateExpressionV2(Tree)
            if expresion == None:
                return None, None,None
            y_predict  = [ (self.evaluar_expresion(expresion, x)) for x in self.X_true ]
            #Verificamos si la funcion es valida
            isValida = not any(x ==  float('inf') for x in y_predict)
            mse = self.calcular_ecm(self.y_true, y_predict)
            return expresion,y_predict,mse,isValida
        except Exception as e:
            # print(f"Error generatingInfo {e}")
            return expresion,y_predict,mse,False

    def ListBuilld(self,individuo):
        return build(individuo)

    def generate_random_expression(self):
        # # return [ random.choice(self.operators + self.variables  + self.constants + self.operators +self.functions) for _ in range(size)]
        order = ['operator', 
                 'operator','operator', 
                 'operator', 'operator',
                 'operator' ,'operator',
                 'constant', 'variable' ,
                 'constant', 'constant' ,
                 'constant','variable',
                 'constant','variable']
        expression = []
        for item in order:
            if item == 'operator':
                expression.append(random.choice(self.operators))
            elif item == 'variable':
                expression.append(random.choice(self.variables))
            elif item == 'constant':
                expression.append(random.choice(self.constants))
            elif item == 'function':
                expression.append(random.choice(self.functions))
            else:
                expression.append(None)
        return expression
    
    def calcular_ecm(expresion, y_true,y_pred):
        try:
            valid_pares = [(true, pred) for true, pred in zip(y_true, y_pred) if true is not None and pred is not None]
        
        # Desempaquetar los pares válidos en arreglos NumPy
            y_true = np.array([true for true, pred in valid_pares])
            y_pred = np.array([pred for true, pred in valid_pares])
        
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            return  round( np.mean((y_true - y_pred) ** 2),4)
        except Exception as e:
            print("Error calcular_ecm" + str(e))
            return 1.234e6 
    
    def MSE(y, y_):
        try:
            m = y.shape[0]
            mse = 0
            for i in range(m):
                mse += (y[i] - y_[i]) ** 2
            return round(mse / m, 2)
        except Exception as e:
            print("Error MSE" + str(e))
            return 1.234e6 
    def seleccionarPadre(self,posiblePadres,poblacion):
        seleccion = random.randint(0, len(posiblePadres)-1)
        padres = posiblePadres[seleccion]
        #Realizando la cruza
    
        p1 = poblacion[padres[0]]
        p2 = poblacion[padres[1]]
        return p1,p2
    
    def generateGeneration(self,poblacion):
        newGeneracion = []
        try:
            #Obtenemos los padres.... Revisar como realizar ruleta u otro metodo
            #Generamos permuitacion
            pos = list(range(0, len(poblacion)))
            combinacion = list(combinations(pos, 2))
            for i in range(len(poblacion)):
                try:
                    posiblePadres = [j for j in combinacion if i not in j]
                    # seleccion = random.randint(0, len(posiblePadres)-1)
                    # padres = posiblePadres[seleccion]
                    # #Realizando la cruza
                    # p1 = poblacion[padres[0]]
                    # p2 = poblacion[padres[1]]
                    p1, p2 = self.seleccionarPadre(posiblePadres,poblacion)
                    #Aqui valido si la cruza se realiza sobre los mismos tipos de nodods
                    hijo1 = copy.deepcopy(p1["tree"])
                    hijo2 = copy.deepcopy(p2["tree"])
                    node1 = None
                    node2 = None
                    isTypeEquals = True
                    iteracionesB = 0
                    while isTypeEquals:
                        # print(f"Buscando Nodos Iguales {iteracionesB}")
                        node1 = self.seleccionNode(hijo1)
                        node2 = self.seleccionNode(hijo2)
                        isTypeEquals = True
                        # if iteracionesB <20:
                        #     print("Limite superado eleccion de nuevos padres")
                        #     p1, p2 = self.seleccionarPadre(posiblePadres,poblacion)
                        #     #Aqui valido si la cruza se realiza sobre los mismos tipos de nodods
                        #     hijo1 = copy.deepcopy(p1["tree"])
                        #     hijo2 = copy.deepcopy(p2["tree"])
                        #     iteracionesB = 0
                        iteracionesB += 1
                        if node1 != None and node2 != None:
                            isTypeEquals = self.validarTipo(node1[1].value,node2[1].value)
                            

                    """Realizando la cruza"""
                    try:
                        hijo1[node1[0]] = node2[1]
                        hijo2[node2[0]] = node1[1]     
                    except Exception as e:
                        print(f"Error {e} ")

                    """Falta validar la profundidad"""
                    expresion,y_predict,mse,isValida =self.generateInfo(hijo1)
                    expresion2,y_predict2,mse2,isValida2 =self.generateInfo(hijo2)

                    elMejor = copy.deepcopy(hijo1)
                    if mse2 ==  None and mse != None :
                        newGeneracion.append({"tree":elMejor, "expresion":expresion,"y_predict":y_predict,"mse":mse,"isValida":isValida})
                        continue

                    if mse2 !=  None and mse == None :
                        elMejor = copy.deepcopy(hijo2)
                        newGeneracion.append({"tree":elMejor, "expresion":expresion2,"y_predict":y_predict2,"mse":mse2,"isValida":isValida2})
                        continue
                        
                    if mse2 ==  None and mse == None :
                        newGeneracion.append(p1)
                        continue

                    if mse2<mse:
                        elMejor = copy.deepcopy(hijo2)
                        expresion = expresion2
                        y_predict = y_predict2
                        isValida = isValida2
                        mse = mse2
                    """Realizando la Muta"""
                    elMejor =self.generateMuta(elMejor)   
                    newGeneracion.append({"tree":elMejor, "expresion":expresion,"y_predict":y_predict,"mse":mse,"isValida":isValida})
                except Exception as e:
                    continue
                    print("Error en generateGeneration: " + str(e) + str(i))
            newGeneracion = sorted(newGeneracion, key=lambda x: np.inf if x['mse'] is None or np.isnan(x['mse']) else x['mse'])

            # newGeneracion = sorted(newGeneracion, key=lambda x: x['mse'] if x['mse'] is not None else float('inf'))
            return newGeneracion
        except Exception as e:
            # print("Error en generateGeneration: " + str(e))
            return newGeneracion

    def seleccionNode(self,seleccionTree):
        try:
            opciones =  [ indice for indice, elemento in enumerate(seleccionTree.values) if elemento is not None and indice != 0] 
            randomOPcion = random.choice(opciones)
            opcion = (randomOPcion, seleccionTree[randomOPcion])
            return opcion
        except Exception as e:
            # print(f"Error seleccionNode {e}")
            return None
        
    def generateMuta(self,Tree):
        try:
            # print("Iniciamos la Muta Este es el mejor")
            # mutaTree = copy.deepcopy(Tree)
            # print(Tree)
            mutaTree = copy.deepcopy(Tree)
            value = "X"
            nodeS = None
            while value == "X":
                nodeS = self.seleccionNode(mutaTree)
                value = nodeS[1].value

            temp = None        
            #Verficamos si esl valor es un numero un operador o una funcion
            if value.isdigit():
                temp = ["1","1"]
            elif value in self.operators:
                temp = self.operators
            elif value in self.functions:
                temp = self.functions
            else:
                print("El value no encontraod")
            #if value != "X":
            posibles = [temp[i] for i in range(len(temp)) if temp[i] != value]
            nuevoValue = random.choice(posibles)
            mutaTree[nodeS[0]].value = nuevoValue
            return mutaTree
        except Exception as e:
            # print("Error en generar muta: " + str(e))
            return mutaTree
    
    def validarTipo(self, valueN1, valueN2):
        # print(valueN1,valueN2)
        if valueN1.isdigit() and valueN2.isdigit() or  valueN1 == "X" and valueN2.isdigit()  or   valueN1.isdigit() and valueN2 == "X"  :
            return False
        if  valueN1 in self.operators and  valueN2 in self.operators:
            return False
        if valueN1 in self.functions and valueN2 in self.functions:
            return False
        if valueN1 == valueN2:
            return True
        return True

    def evaluar_expresion(self, expresion,X):
        try:
            return round(eval(expresion, {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,"X": X , 'log' :math.log}),4 )
        except ZeroDivisionError:
            return 1.234e6
        except Exception as e:
            return 1.234e6
