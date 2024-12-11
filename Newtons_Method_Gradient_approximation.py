import numpy as np

def newton_method_numerical(f, x0, epsilon, Nmax, h=1e-6):
    """
    Méthode de Newton avec approximations numériques pour minimiser une fonction réelle f(x).
    """
    def gradient_approximation(f, x, h):
        """Calcule le gradient approximatif par différences finies centrées."""
        return (f(x + h) - f(x - h)) / (2 * h)

    def hessian_approximation(f, x, h):
        """Calcule la matrice Hessienne approximative (dérivée seconde) par différences finies."""
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

    k = 0
    xk = np.array(x0, dtype=float)

    while k < Nmax:
        grad = gradient_approximation(f, xk, h)
        hess = hessian_approximation(f, xk, h)

        if abs(grad) <= epsilon:
            print(f"Convergence atteinte: f'(x{k}) = {grad:.10f}")
            break

        if hess > 0:  # Vérifie si la dérivée seconde est définie positive
            xk_new = xk - grad / hess
            print(
                f"Étape {k}: nouvelle approximation x{k} = {xk}, "
                f"x{k} = {xk_new}, f'(x{k}) = {grad:.10f}, f''(x{k}) = {hess:.10f}"
            )
            xk = xk_new
        else:
            print("La dérivée seconde (Hessienne) n'est pas définie positive.")
            return None

        k += 1

    return xk

# Exemple d'utilisation
def fonction(x):
    return 1 - 1 / (5 * x[0]**2 - 6 * x[0] + 5)

# Configuration de l'algorithme
x0 = [0.5]  # Approximation initiale
epsilon = 1e-6  # Tolérance
Nmax = 100  # Nombre maximal d'itérations

# Exécution de l'algorithme
minimum = newton_method_numerical(fonction, x0, epsilon, Nmax)
print("Approximation du minimum local:", minimum)
