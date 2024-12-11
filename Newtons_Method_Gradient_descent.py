import numpy as np

# Définition de la fonction
def f(x):
    return 1 - 1 / (5 * x[0]**2 - 6 * x[0] + 5)

def derivee_numerique(f, x, h=1e-6):
    """
    Calcule la dérivée numérique d'une fonction en utilisant les différences finies.
    """
    return (f(x + h) - f(x - h)) / (2 * h)
    
def derivee_seconde_numerique(f, x, h=1e-6):
    """
    Calcule la dérivée seconde d'une fonction en utilisant les différences finies.
    
    Arguments:
    - f : fonction à dériver
    - x : point où calculer la dérivée
    - h : pas d'approximation (par défaut 1e-6)
    
    Retourne:
    - La valeur de la dérivée seconde au point x.
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


# Calcul du gradient (dérivée première) avec des différences finies
def grad_f(x, h=1e-6):
    return (f(x + h) - f(x)) / h

# Calcul du Hessien (dérivée seconde) avec des différences finies
def hess_f(x, h=1e-6):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

# Méthode de Newton pour l'optimisation de la fonction
def newton_method(f, grad_f, hess_f, x0, epsilon, Nmax):
    k = 0
    xk = x0  # Point initial
    
    while k < Nmax:
        # Calcul du gradient et du Hessien
        grad = grad_f(xk)
        hess = hess_f(xk)
        
        # Vérification si le Hessien est défini positif
        if hess > 0:
            # Mise à jour du point avec la méthode de Newton
            xk_new = xk - grad / hess
            print(f"Étape {k}: nouvelle approximation x{k} = {xk} , x{k}+1 = {xk_new} , f'(x{k}+1) = {derivee_numerique(f, xk_new):.10f} , f''(x{k}+1) = {derivee_seconde_numerique(f, xk_new):.10f}")
        else:
            print("Le Hessien n'est pas défini positif.")
            return None
        
        # Vérification de la convergence en fonction du gradient
        if abs(grad) < epsilon:
            print(f" Le gradient est suffisamment petit.")
            break
        
        # Mise à jour du point
        xk = xk_new
        k += 1
    
    print(f"Nombre total d'étapes: {k}")
    return xk

# Paramètres de la méthode de Newton
x0 = np.array([0.5000000000])  # Point initial
epsilon = 1e-6  # Tolérance
Nmax = 100  # Nombre maximal d'itérations

# Application de la méthode de Newton
minimum = newton_method(f, grad_f, hess_f, x0, epsilon, Nmax)

print("Approximation du minimum local:", minimum)
