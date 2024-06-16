from binarytree import Node,build
import random
def ManuallyBuild():
    #Raiz
    root = Node("+")
    #Nivel 1
    root.left = Node("-")
    root.right = Node("*")
    #Nivel 2
    root.left.left = Node("2.2")
    root.left.right = Node("/")

    root.right.left = Node("7")
    root.right.right = Node("cos")

    #Nivel 3
    root.left.right.left = Node("X")
    root.left.right.right = Node("11")

    root.right.right.left = Node("Y")
    print(root)

    print(f'Imprimiendo in order {root.inorder}' )

def ListBuilld(individuo):
    return build(individuo)

def seleccionNode(Tree):
    opciones =  [ (indice, elemento) for indice, elemento in enumerate(Tree) if elemento is not None ] 
    return random.choice(opciones)

def generateArboles():
    E1 =["+","-","*","8.6","/","15","Y",None,None,"X","11",None,None,"sin"]
    E2 = ["*", "+", "-", "X", "7", "8.8", "/", "Y", "11", None, None, None, None, None, "cos"]
    E3 = ["-", "-", "*", "2.2", "/", "7", "X", "11", "cos", "Y", None, None, None, None, None]
    E3 = ["-", "-", "*", "2.2", "/", "7", "X", "11", "cos", "Y", None, None, None, None, None]

    E1Tree = ListBuilld(E1)
    E2Tree = ListBuilld(E2)
    E3Tree = ListBuilld(E3)
    return E1Tree, E2Tree, E3Tree

def generateExpression(Tree):
    # print(f'Imprimiendo in order {Tree.inorder}' )
    expression = [ Tree.inorder[i].values[0] for i in range(len(Tree.inorder))]
    return "".join(expression)


def main():
    E1Tree, E2Tree, E3Tree = generateArboles()
    print(10*"#"+"Inicio de la poblacion"+10*"#")
    print(f"E1 {E1Tree}")
    expresion = generateExpression(E1Tree)
    print(expresion)
    
    print(f"E2 {E2Tree}")
    print(f"E3 {E3Tree}")
    print(40*"#")
    print(10*"#"+"Seleccion"+10*"#")

    """
        E1  = E2 + E3
        E2  = E1 + E3
        E3  = E1 + E2
    """

    nodo1 = seleccionNode(E2Tree)
    nodo2 = seleccionNode(E3Tree)

    print(f"Del arbol E2 Nodo seleccionado {nodo1}")
    print(f"Del arbol E3 Nodo seleccionado {nodo2}")
    # print(f"E1 {E1Tree}")
    # print(f"E2 {E2Tree}")
    # print(f"E3 {E3Tree}")



# main()


def evaluar_expresion(expresion):
    try:
        resultado = eval(expresion)
        return resultado
    except ZeroDivisionError:
        return "Error: División por cero"
    except:
        return "Error: Expresión inválida"
    
# Lista de nodos en orden de nivel (level-order traversal)
# nodes = [
#         '+',  
#         '*', '-',  
#         '*', '+',  
#         '+', '/', 
#         '5', '10',
#         '15', '20',
#         '2', '3',
#         '6', '7'
#         ]

# tree = build(nodes)
# print(tree)
# print(generateExpression(tree))
# resultado = evaluar_expresion(generateExpression(tree))
# tree[1].value = "e"
# print(tree)
# tree.max_leaf_depth
# print("")

