import numpy as np
import sympy as sp

# Function definition
def f(x):
    """
    Function to optimize, takes a list/array x as input.
    Example: f(x) = 1 - 1 / (5 * x1**2 - 6 * x1 + 5)
    """
    return 1 - 1 / (5 * x[0]**2 - 6 * x[0] + 5)

def compute_gradient(func, variables):
    """
    Automatically computes the gradient of a multivariable function.

    :param func: Symbolic function defined using symbolic variables
    :param variables: List of symbolic variables
    :return: Gradient as a list of partial derivatives
    """
    return [sp.diff(func, var) for var in variables]

def line_search(func, x_k, d_k, variables):
    """
    Performs a line search to determine the optimal step size alpha_k.

    :param func: Symbolic function to optimize
    :param x_k: Current point (array)
    :param d_k: Descent direction (array)
    :param variables: List of symbolic variables
    :return: Optimal step size alpha_k
    """
    # Define alpha as a symbolic variable
    alpha = sp.Symbol('alpha')
    
    # Substitute x_k + alpha * d_k into the function
    x_new = [x_k[i] + alpha * d_k[i] for i in range(len(x_k))]
    func_alpha = func.subs({var: x_new[i] for i, var in enumerate(variables)})
    
    # Compute the derivative of func_alpha with respect to alpha
    func_alpha_prime = sp.diff(func_alpha, alpha)
    
    # Solve func_alpha_prime = 0 for alpha
    alpha_sol = sp.solvers.solve(func_alpha_prime, alpha)
    
    # Ensure the solution is valid (real and positive)
    alpha_k = [float(a) for a in alpha_sol if sp.re(a) > 0]
    
    return alpha_k[0] if alpha_k else 1e-3  # Default to a small step size if no valid solution

def gradient_descent_with_line_search(f_sym, variables, x0, epsilon, max_iter=1000):
    """
    Gradient descent method with line search for step size determination.

    :param f_sym: Symbolic function to optimize
    :param variables: List of symbolic variables
    :param x0: Initial point (Numpy array)
    :param epsilon: Tolerance for convergence
    :param max_iter: Maximum number of iterations
    :return: Optimal solution found
    """
    # Compute the symbolic gradient
    gradient_sym = compute_gradient(f_sym, variables)
    gradient_funcs = [sp.lambdify(variables, grad, 'numpy') for grad in gradient_sym]
    
    # Initialization
    x_k = np.array(x0, dtype=float)
    
    for k in range(max_iter):
        # Compute the gradient at the current point
        grad_values = np.array([grad_func(*x_k) for grad_func in gradient_funcs])
        
        # Check the norm of the gradient for stopping criterion
        grad_norm = np.linalg.norm(grad_values)
        print( f"Étape {k}: nouvelle approximation x{k} = {x_k}" )
        if grad_norm <= epsilon:
            print(f"Convergence reached at iteration {k+1}")
            return x_k
        
        # Descent direction is the negative gradient
        d_k = -grad_values
        
        # Line search to find optimal step size
        alpha_k = line_search(f_sym, x_k, d_k, variables)
        
        # Update the solution
        x_k = x_k + alpha_k * d_k
        
        print(f"Iteration {k}: x{k} = {x_k}, alpha_k = {alpha_k}, grad_norm = {grad_norm}")
    
    print("Maximum number of iterations reached.")
    return x_k

# Define symbolic variables
x1_sym = sp.Symbol('x1')
f_sym = f([x1_sym])  # Convert the function into symbolic form

# Initialization and parameters
x0 = [2.0]  # Initial point
epsilon = 1e-6  # Tolerance
max_iter = 1000  # Maximum number of iterations

# Call the gradient descent method with line search
solution = gradient_descent_with_line_search(f_sym, [x1_sym], x0, epsilon, max_iter)
print("Optimal solution found:", solution)

# Verify with the original function
f_func = sp.lambdify(x1_sym, f_sym, 'numpy')
print("Value of the function at the optimal solution:", f_func(solution[0]))



# min alpha (f_sym, x_k, d_k, variables)

#Fonction gradient_descent_with_line_search(f_sym, variables, x0, epsilon, max_iter):
#    Initialiser gradient_sym ← gradient_symbolique(f_sym)
#    Convertir gradient_sym en fonctions évaluables
#    
#    x_k ← x0  // Point initial
#    
#    Pour k ← 0 à max_iter faire :
#        Calculer grad_values ← évaluer_gradient(gradient_sym, x_k)
#        Si norme(grad_values) ≤ epsilon :
#            Retourner x_k  // Convergence atteinte
#        
#        d_k ← -grad_values  // Direction de descente
#        alpha_k ← recherche_linéaire(f_sym, x_k, d_k, variables)
#        
#        Mettre à jour x_k ← x_k + alpha_k * d_k
#        
#        Afficher itération courante : x_k, alpha_k, norme(grad_values)
#    
#    Afficher "Nombre maximal d'itérations atteint"
#    Retourner x_k
