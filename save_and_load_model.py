#2 functions to save the trained model W1,B1,W2,B2
#using np.savez to save these numpy arrays in format .npz 
#load model first checks if the file already exists or not in order to create a new model from scratch if necessary

import os

def save_model(filename,W1,B1,W2,B2,epsilon,nb_iterations):
    np.savez(filename,W1=W1,B1=B1,W2=W2,B2=B2,epsilon=epsilon,nb_iterations=nb_iterations)# param=param  sert à étiqueter les données pour faciliter l'ouverture
                                                                                            # le fichier sera enregistré au format .npz qui est un fichier binaire compressé
def load_model(filename):
    # Si le fichier n'existe pas, on renvoie None pour dire "Initialise à zéro"
    if not os.path.exists(filename):
        print(f" Aucun fichier '{filename}' trouvé. Démarrage d'un nouvel entraînement.")
        return None
    
    # On charge le fichier
    data = np.load(filename)
    
    # On extrait les variables
    W1 = data['W1']
    B1 = data['B1']
    W2 = data['W2']
    B2 = data['B2']
    epsilon = float(data['epsilon']) # On convertit en float standard
    nb_iterations = data['nb_iterations']
    
    print(f" Modèle chargé depuis '{filename}' ! Reprise avec nb_iterations = {nb_iterations} et {epsilon}")
    return W1, B1, W2, B2, epsilon,nb_iterations


def main():
    
    # Paramètres dimensionnels
    n_x = 11
    n_h = 256
    n_y = 3
    
    filename = "snake_model.npz" # Nom du fichier de sauvegarde

    # 1. TENTATIVE DE CHARGEMENT
    loaded_data = load_model(filename)

    if loaded_data:
        # Si on a trouvé un fichier, on récupère les poids entraînés
        W1, B1, W2, B2, epsilon,nb_iterations = loaded_data
    else:
        # Sinon, on initialise tout à neuf (He initialization)
        W1 = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)
        B1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * np.sqrt(2/n_h)
        B2 = np.zeros((n_y, 1))
        epsilon = 1.0 # On commence à 100% exploration
        nb_iterations=0
    
    
    # Test
    resultats = []
    epsilon=1
    print("Démarrage training...")
    iterations=1000
    for i in range(iterations):
        W1, B1, W2, B2,c = training_session(epsilon, W1, B1, W2, B2)
        epsilon=max(0.001,epsilon*0.99)

        ##stockage des pommes mangées pour chaque partie
        resultats.append(c)

    nb_iterations+=iterations
    print("Training terminé sans erreur !",nb_iterations)



    ##Affichage lisse en rouge et brut en gris
    # 1. Calcul de la moyenne mobile (Moving Average)
    window_size = 50 # On fait la moyenne sur les 50 dernières parties
    if len(resultats) >= window_size:
        # np.convolve permet de lisser la courbe
        moyenne_mobile = np.convolve(resultats, np.ones(window_size)/window_size, mode='valid')
    else:
        moyenne_mobile = resultats # Fallback si pas assez de données

    plt.figure(figsize=(12, 8)) # Agrandir la figure
    
    # 2. Afficher les scores bruts en gris clair (pour voir la variance)
    plt.plot(resultats, label='Score par partie', color='lightgray', alpha=0.6)
    
    # 3. Afficher la moyenne mobile en rouge (pour voir la progression)
    # On décale l'axe X pour qu'il s'aligne bien (car la moyenne commence après 'window_size' parties)
    x_axis = np.arange(len(moyenne_mobile)) + window_size - 1
    plt.plot(x_axis, moyenne_mobile, label=f'Moyenne mobile ({window_size} parties)', color='red', linewidth=2)

    # 4. Labels et Titres
    plt.title("Progression de l'apprentissage du Snake")
    plt.xlabel("Numéro de l'entraînement")
    plt.ylabel("Nombre de pommes mangées")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5) # Une grille aide à lire les valeurs
    
    plt.show()

    save_model(filename,W1,B1,W2,B2,epsilon,nb_iterations)
