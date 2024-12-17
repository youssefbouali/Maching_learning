import numpy as np
import sympy as sp

# Définition de la fonction
def f(x):
    return 1 - 1 / (5 * x[0]**2 - 6 * x[0] + 5)

# Conversion automatique en fonction symbolique
x_sym = sp.Symbol('x')  # Variable symbolique
f_sym = f([x_sym])  # Générer l'expression symbolique à partir de la fonction

# Calcul automatique des dérivées analytiques
f_prime_sym = sp.diff(f_sym, x_sym)  # Première dérivée
f_seconde_sym = sp.diff(f_prime_sym, x_sym)  # Seconde dérivée

# Transformation des expressions symboliques en fonctions Python
f_prime = sp.lambdify(x_sym, f_prime_sym, 'numpy')
f_seconde = sp.lambdify(x_sym, f_seconde_sym, 'numpy')

# Méthode de Newton pour l'optimisation de la fonction
def newton_method(f, f_prime, f_seconde, x0, epsilon, Nmax):
    k = 0
    xk = x0  # Point initial

    while k < Nmax:
        # Calcul du gradient et du Hessien
        grad = f_prime(xk)
        hess = f_seconde(xk)
        
        # Vérification si le Hessien est défini positif
        if hess > 0:
            # Mise à jour du point avec la méthode de Newton
            xk_new = xk - grad / hess
            print(f"Étape {k}: x{k} = {xk:.10f}, x{k+1} = {xk_new:.10f}, f'(x) = {grad:.10f}, f''(x) = {hess:.10f}")
        else:
            print("Le Hessien n'est pas défini positif.")
            return None
        
        # Vérification de la convergence en fonction du gradient
        if abs(grad) < epsilon:
            print("Le gradient est suffisamment petit.")
            break
        
        # Mise à jour du point
        xk = xk_new
        k += 1

    print(f"Nombre total d'étapes: {k}")
    return xk

# Paramètres de la méthode de Newton
x0 = 0.5  # Point initial
epsilon = 1e-6  # Tolérance
Nmax = 100  # Nombre maximal d'itérations

# Application de la méthode de Newton
minimum = newton_method(f, f_prime, f_seconde, x0, epsilon, Nmax)

print("Approximation du minimum local:", minimum)
