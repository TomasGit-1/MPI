from binarytree import Node
import random
import math

class TreeC:
    def __init__(self,operators=[],functions=[]):
        self.operadores = operators
        self.funciones = functions
        
    def build(self, profundidad_maxima):
        if profundidad_maxima == 0:
            return Node(random.choice([str(1), 'X']))
        else:
            raiz = Node(random.choice(self.operadores))
            raiz.left = self.build(profundidad_maxima - 1)
            if random.random() < 0.5:  
                # 50% de probabilidad de elegir un operador
                raiz = Node(random.choice(self.operadores))
                raiz.left = self.build(profundidad_maxima - 1)
                raiz.right = self.build(profundidad_maxima - 1)
            else:  
                # Solo un hijo para las funciones unarias
                raiz = Node(random.choice( self.funciones))
                raiz.right = self.build(profundidad_maxima - 1)  
            return raiz
    
    def generateExpressionV2(self,tree):
        try:
            if tree is None:
                return ''
        
            if tree.left is None and tree.right is None:
                return tree.value
            
            expresion_izquierda = self.generateExpressionV2(tree.left)
            expresion_derecha = self.generateExpressionV2(tree.right)
            if tree.value in self.operadores:
                expresion_actual = '(' + str(expresion_izquierda) + tree.value + str(expresion_derecha) + ')'
                return expresion_actual
            elif tree.value.lower() in self.funciones:
                expresion_actual = tree.value.lower() + '(' + expresion_derecha + ')'
                return expresion_actual
            else:
                return ""
        except Exception as e:
            print("Error: {}".format(e))
            expresion_actual = ""
    
    def generateExpressionV1(self, Tree):
        expression = [ Tree.inorder[i].values[0] for i in range(len(Tree.inorder))]
        detectarC3 = 0
        expresionFull = ""
        for i in range(len(expression)):
            if detectarC3 == 0:
                expresionFull += "("
            if detectarC3 == 3:
                expresionFull += ")"
                expresionFull += expression[i] 
                detectarC3 = 0
            else:
                expresionFull += expression[i] 
                detectarC3 +=1
        return expresionFull+")"

    def cortar_Tree(self, Tree, profundidad_deseada, profundidad_actual=0):
        if Tree is None or profundidad_actual == profundidad_deseada:
            return None        
        Tree.izquierda = self.cortar_Tree(Tree.izquierda, profundidad_deseada, profundidad_actual + 1)
        Tree.derecha = self.cortar_Tree(Tree.derecha, profundidad_deseada, profundidad_actual + 1)
        return Tree

