def derivee_numerique(f, x, h=1e-6):
    """
    Calcule la dérivée numérique d'une fonction en utilisant les différences finies.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    """
    Applique la méthode de Newton-Raphson pour trouver une racine.
    f : fonction
    x0 : estimation initiale
    tol : tolérance d'erreur
    max_iter : nombre maximum d'itérations
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = derivee_numerique(f, x)
        print(f"x{i+1} : {x} - f(x) = {f(x)}")
        if abs(fx) < tol:
            print(f"Racine trouvée après {i+1} itérations")
            return x
        if fpx == 0:
            raise ValueError("La dérivée est nulle, la méthode échoue")
        x -= fx / fpx
    print(f"Approximation après {max_iter} itérations : {x}")
    return x




# Définir la fonction
f = lambda x: x**2 - 2

# Utiliser la méthode
x0 = 1  # estimation initiale
newton_raphson(f, x0)
