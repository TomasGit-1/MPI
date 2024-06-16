from binarytree import Node
import random
import math

class TreeC:
    def __init__(self, limite=100):
        self.limite = limite
        self.operadores = ['+', '-', '*', '/']
        self.funciones = ['sin', 'cos']
        
    def build(self, profundidad_maxima):
        if profundidad_maxima == 0:
            # Cuando llegamos a la profundidad máxima, creamos una hoja con un número aleatorio
            return Node(random.choice([str(random.randint(1, self.limite)), 'X']))
        else:
            # En niveles internos, seleccionamos un operador o función trigonométrica
            # raiz = Node(random.choice(['+', '-', '*', '/', 'sin', 'cos','log']))
            # if raiz.value in ['sin', 'cos', 'tan', 'log']:
            #     # raiz.left = None
            #     raiz.right = self.build(profundidad_maxima - 1)
            # else:
            #     raiz.left = self.build(profundidad_maxima - 1)
            #     raiz.right = self.build(profundidad_maxima - 1)
              # En niveles internos, la raíz será siempre un operador
            raiz = Node(random.choice(self.operadores))
            raiz.left = self.build(profundidad_maxima - 1)
            if random.random() < 0.5:  # 50% de probabilidad de elegir un operador
                raiz = Node(random.choice(self.operadores))
                raiz.left = self.build(profundidad_maxima - 1)
                raiz.right = self.build(profundidad_maxima - 1)
            else:  # 50% de probabilidad de elegir una función
                raiz = Node(random.choice( self.funciones))
                raiz.right = self.build(profundidad_maxima - 1)  # Solo un hijo para las funciones unarias
        
            return raiz
    
    def generateExpressionV2(self,tree):
        try:
            if tree is None:
                return ''
        
            if tree.left is None and tree.right is None:
                return tree.value
            
            expresion_izquierda = self.generateExpressionV2(tree.left)
            expresion_derecha = self.generateExpressionV2(tree.right)
            
            if tree.value in ('+', '-', '*', '/'):
                expresion_actual = '(' + str(expresion_izquierda) + tree.value + str(expresion_derecha) + ')'
                return expresion_actual
            elif tree.value.lower() in ('sin', 'cos', 'tan','log'):
                expresion_actual = tree.value.lower() + '(' + expresion_derecha + ')'
                return expresion_actual
            else:
                # Operador no válido, puede ser extendido si hay más operadores
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

