import numpy as np
import sympy as sp

# Fonction à optimiser
def f(x):
    """
    Fonction à optimiser. Exemple : f(x) = 1 - 1 / (5 * x1**2 - 6 * x1 + 5)
    """
    return 1 - 1 / (5 * x[0]**2 - 6 * x[0] + 5)

def compute_gradient(func, variables):
    """
    Calcule automatiquement le gradient d'une fonction multivariable.

    :param func: Fonction symbolique définie avec des variables symboliques
    :param variables: Liste de variables symboliques
    :return: Gradient sous forme de liste des dérivées partielles
    """
    return [sp.diff(func, var) for var in variables]

def evaluate_gradient(gradient, values):
    """
    Évalue le gradient à un point donné.

    :param gradient: Gradient symbolique
    :param values: Liste de valeurs pour les variables
    :return: Gradient évalué
    """
    return np.array([float(g.evalf(subs={x: v for x, v in zip(variables, values)})) for g in gradient])

# Fonction de descente du gradient
def descente_gradient(f, grad_f, x0, epsilon, N, c=1e-4):
    """
    Algorithme de descente du gradient avec calcul du pas selon la suite géométrique.

    Paramètres:
    f : fonction objectif
    grad_f : fonction du gradient de f
    x0 : point de départ (vecteur numpy)
    epsilon : critère de convergence (scalair)
    N : nombre maximal d'itérations
    c : paramètre de Armijo (par défaut 1e-4)

    Retourne :
    xk : point optimal trouvé
    fk : valeur de la fonction au point optimal
    """
    xk = x0
    k = 0

    while k <= N:
        # Calcul du gradient au point xk
        grad = grad_f(xk)
        
        print( f"Étape {k}: nouvelle approximation x{k} = {xk}" )
        # Direction de descente
        dk = -grad
        
        # Vérification du critère de convergence
        if np.linalg.norm(dk) <= epsilon:
            break
        
        # Recherche du pas alpha_k
        alpha_k = 1
        while f(xk + alpha_k * dk) >= f(xk) + c * alpha_k * np.dot(grad, dk):
            alpha_k *= 0.5  # Réduction du pas
        
        # Mise à jour du point
        xk = xk + alpha_k * dk
        
        # Incrémentation du compteur d'itérations
        k += 1
    
    # Retour du résultat
    return xk, f(xk)


# Définition des variables symboliques pour la fonction
x1 = sp.symbols('x1')

# Fonction symbolique
func = 1 - 1 / (5 * x1**2 - 6 * x1 + 5)

# Calcul du gradient symbolique
variables = [x1]
gradient = compute_gradient(func, variables)

# Exemple d'utilisation avec un point initial
x0 = np.array([1.0])  # Point de départ
epsilon = 1e-6  # Critère de convergence
N = 100  # Nombre maximal d'itérations

# Exécution de l'algorithme avec le calcul automatique du gradient
x_opt, f_opt = descente_gradient(f, lambda x: evaluate_gradient(gradient, x), x0, epsilon, N)

print("Point optimal:", x_opt)
print("Valeur optimale:", f_opt)



#critère d'Armijo