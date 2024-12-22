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
warnings.filterwarnings("error", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.ComplexWarning)
class PGenetica:
    def __init__(self,log, X_true,y_true, operators=[],functions=[]):
        self.log = log
        self.operators = operators
        self.variables = ["X"]
        self.functions = functions
        self.constants = [1]
        self.X_true = X_true
        self.y_true = y_true
        self.objTree = TreeC(self.operators, self.functions)
        self.errorMseMax = 1.5e7

    def generatePoblacionAleatoria(self, poblacionSize = 4, profundidad=4):
        self.log.debug("Generando poblacion aleatoria")
        poblacion = []
        i=0
        while i < poblacionSize:
            Tree = self.objTree.build(profundidad)
            expresion,y_predict,mse,isValida = self.generateInfo(Tree)    
            poblacion.append({"tree":Tree, "expresion":expresion,"y_predict":y_predict,"mse":mse, "isValida":isValida})
            i+=1
        self.log.debug(f"Poblacion generada ")
        return poblacion
    
    def ordenarPoblacion(self, poblacion):
        ordenada = sorted(poblacion, key=lambda x: x['mse'])
        return ordenada

    
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
            if np.isnan(mse):
                mse = self.errorMseMax
            return expresion,y_predict,mse,isValida
        except Exception as e:
            # print(f"Error generatingInfo {e}")
            return expresion,y_predict,self.errorMseMax,False

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
    
    def calcular_ecm(self, y_true,y_pred):
        try:
            return round(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) ** 2),4)
        except Exception as e:
            print("Error calcular_ecm " + str(e))
            return self.errorMseMax
 
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
            self.log.debug("Obtenemos los padres....Generamos Combinacion ")
            #Generamos permuitacion
            pos = list(range(0, len(poblacion)))
            combinacion = list(combinations(pos, 2))
            for i in range(len(poblacion)):
                self.log.debug(f"Inidivid {i}")
                try:
                    posiblePadres = [j for j in combinacion if i not in j]
                    p1, p2 = self.seleccionarPadre(posiblePadres,poblacion)
                    #Aqui valido si la cruza se realiza sobre los mismos tipos de nodods
                    hijo1 = copy.deepcopy(p1["tree"])
                    hijo2 = copy.deepcopy(p2["tree"])
                    node1 = None
                    node2 = None
                    self.log.debug("Seleccionado hijos")
                    node1 = self.seleccionNode(hijo1)
                    node2 = self.seleccionNode(hijo2)

                    """Realizando la cruza"""
                    try:
                        self.log.debug("generando cruza")
                        hijo1[node1[0]] = node2[1]
                        hijo2[node2[0]] = node1[1]     
                    except Exception as e:
                        self.log.error(f"Error realizando cruza {e} ")

                    self.log.debug("Eligiendo el mejor de los hijos")
                    expresion,y_predict,mse,isValida =self.generateInfo(hijo1)
                    expresion2,y_predict2,mse2,isValida2 =self.generateInfo(hijo2)
                    elMejor = copy.deepcopy(hijo1)
                    if mse2<mse:
                        elMejor = copy.deepcopy(hijo2)
                        expresion = expresion2
                        y_predict = y_predict2
                        isValida = isValida2
                        mse = mse2
                    """Realizando la Muta"""
                    elMejor =self.generateMuta(elMejor)   
                    expresion,y_predict,mse,isValida = self.generateInfo(elMejor)
                    newGeneracion.append({"tree":elMejor, "expresion":expresion,"y_predict":y_predict,"mse":mse,"isValida":isValida})
                except Exception as e:
                    print("Error en generateGeneration: " + str(e) + str(i))
                    continue
            newGeneracion = self.ordenarPoblacion(newGeneracion)
            return newGeneracion
        except Exception as e:
            return newGeneracion

    def seleccionNode(self,seleccionTree):
        try:
            opciones =  [ indice for indice, elemento in enumerate(seleccionTree.values) if elemento is not None and indice != 0] 
            randomOPcion = random.choice(opciones)
            opcion = (randomOPcion, seleccionTree[randomOPcion])
            return opcion
        except Exception as e:
            return None
        
    def generateMuta(self,Tree):
        try:
            mutaTree = copy.deepcopy(Tree)
            if mutaTree.size >3:
                value = "X"
                nodeS = None
                while value == "X":
                    nodeS = self.seleccionNode(mutaTree)
                    value = nodeS[1].value
                    if value != "X":
                        break 
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
                posibles = [temp[i] for i in range(len(temp)) if temp[i] != value]
                nuevoValue = random.choice(posibles)
                mutaTree[nodeS[0]].value = nuevoValue
            return mutaTree
        except Exception as e:
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
            return self.errorMseMax
        except Exception as e:
            return self.errorMseMax
