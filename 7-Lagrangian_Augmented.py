import numpy as np

# Example objective function f(x)
def f(x):
    return np.sum(x**2)

# Gradient of the objective function ∇f(x)
def grad_f(x):
    return 2*x

# Example constraint function h(x)
def h(x):
    return np.sum(x) - 1  # Example constraint: x1 + x2 + ... + xn = 1

# Gradient of the constraint function ∇h(x)
def grad_h(x):
    return np.ones_like(x)  # Gradient of sum(x) - 1 is [1, 1, ..., 1]

# Armijo rule for step size (α)
def armijo_rule(x, λ, c, grad_Lc):
    α = 1.0
    β = 0.5
    σ = 1e-4
    
    # Lagrangian augmented function L_c(x, λ, c)
    L_c = lambda x: f(x) + np.dot(λ, h(x)) + 0.5 * c * np.sum(h(x)**2)
    
    # Check the Armijo condition
    while np.sum(L_c(x - α * grad_Lc)) > np.sum(L_c(x)) - σ * α * np.dot(grad_Lc, grad_Lc):
        α *= β  # Reduce α until condition is satisfied
    
    return α


# Update penalty parameter
def increase_penalty(c):
    return c * 2

# Main algorithm
def lagrangian_augmented(x0, λ0, c0, epsilon, max_iter):
    x = x0
    λ = λ0
    c = c0
    k = 0
    
    while k < max_iter:
        # Compute the gradient of the augmented Lagrangian function ∇L_c(x, λ, c)
        grad_Lc = grad_f(x) + λ * grad_h(x) + c * h(x) * grad_h(x)
        
        # Determine the step size α using Armijo rule
        α = armijo_rule(x, λ, c, grad_Lc)
        
        # Update decision variables x
        x = x - α * grad_Lc
        
        # Calculate the constraint violation
        h_val = h(x)
        
        # Update Lagrange multipliers λ
        λ = λ + c * h_val
        
        # Recalculate the gradient of the augmented Lagrangian
        grad_Lc_new = grad_f(x) + λ * grad_h(x) + c * h(x) * grad_h(x)
        
        # Check stopping criterion
        if np.linalg.norm(grad_Lc_new) <= epsilon:
            return x, λ  # Convergence reached
        
        # Optionally update penalty parameter
        c = increase_penalty(c)
        
        # Increment iteration counter
        k += 1
    
    # Maximum iterations reached
    return x, λ

# Example initialization
x0 = np.array([0.1, 0.1])  # Initial guess for the decision variables
λ0 = np.array([0.1, 0.1])  # Initial Lagrange multipliers
c0 = 1.0  # Initial penalty parameter
epsilon = 1e-6  # Convergence tolerance
max_iter = 100  # Maximum number of iterations

# Running the algorithm
x_optimal, λ_optimal = lagrangian_augmented(x0, λ0, c0, epsilon, max_iter)

print("Optimal solution:", x_optimal)
print("Optimal Lagrange multipliers:", λ_optimal)