import numpy as np

def descente_de_gradient_stochastique(f_gradient, x_init, donnees, taux_apprentissage=0.1, max_iterations=50, tolerance=1e-6):
    """
    Algorithme de descente de gradient stochastique (SGD) avec critère de convergence.
    :param f_gradient: Fonction qui calcule le gradient pour un échantillon donné.
    :param x_init: Valeur initiale de x.
    :param donnees: Données d'entraînement (y_i).
    :param taux_apprentissage: Taux d'apprentissage (𝜂).
    :param max_iterations: Nombre maximal d'itérations (N).
    :param tolerance: Seuil de convergence pour arrêter les itérations.
    :return: La valeur finale de x.
    """
    x = x_init  # Initialisation de x avec la valeur donnée
    
    for t in range(max_iterations):
        # Sauvegarder la valeur précédente de x
        x_precedent = x

        # Sélectionner un échantillon aléatoire
        i = np.random.randint(0, len(donnees))
        echantillon = donnees[i]
        
        # Calculer le gradient pour l'échantillon
        grad = f_gradient(x, echantillon)
        
        # Mettre à jour x
        x = x - taux_apprentissage * grad

        # Afficher l'évolution (optionnel)
        print(f"Itération {t+1} : x = {x:.4f} pour x_old={i} y={echantillon}")

        # Vérifier le critère de convergence
        if abs(x - x_precedent) < tolerance:
            print(f"Convergence atteinte à l'itération {t+1}")
            break
    
    return x

# Fonction gradient pour une équation quadratique
def gradient_quadratique(x, echantillon):
    return 2 * (x - echantillon)

# Données d'entraînement (y_i)
donnees = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Initialisation des paramètres
x_initial = 0.0  # Valeur initiale de x
taux_apprentissage = 0.1  # Taux d'apprentissage
max_iterations = 50  # Nombre maximal d'itérations
tolerance = 1e-6  # Seuil de convergence

# Exécuter l'algorithme
x_final = descente_de_gradient_stochastique(gradient_quadratique, x_initial, donnees, taux_apprentissage, max_iterations, tolerance)

print(f"\nValeur finale de x : {x_final:.4f}")
