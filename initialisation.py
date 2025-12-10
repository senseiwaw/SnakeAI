# Architecture
n_x = 11    # Entrée
n_h1 = 256  # Cachée 1
n_h2 = 128  # Cachée 2 (On réduit souvent la taille au fur et à mesure)
n_y = 3     # Sortie
#la fonction non linéaire du modèle est ReLU 
#on procède par initialisation de type Kaiming He (cf : https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/ )
#ici c'est un exemple à deux couches

# --- W1 connecte l'Entrée (11) à Cachée 1 (256) ---
# On divise par n_x (11) car 11 fils arrivent
W1 = np.random.randn(n_h1, n_x) * np.sqrt(2 / n_x) 
B1 = np.zeros((n_h1, 1))

# --- W2 connecte Cachée 1 (256) à Cachée 2 (128) ---
# On divise par n_h1 (256) car 256 fils arrivent
W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2 / n_h1)
B2 = np.zeros((n_h2, 1))

# --- W3 connecte Cachée 2 (128) à Sortie (3) ---
# On divise par n_h2 (128) car 128 fils arrivent
W3 = np.random.randn(n_y, n_h2) * np.sqrt(2 / n_h2)
B3 = np.zeros((n_y, 1))
