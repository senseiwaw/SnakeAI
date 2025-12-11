How to **plot** the data (number of apples eaten by number of games played) ?

using *matplotlib.pyplot* 
the raw data is not useful for a model not yet trained (grey color) 
-> *np.convolve* to make the function smooth (red color)

```python
#fonction main avec affichage
import matplotlib.pyplot as plt

def main():

    
    # Paramètres dimensionnels
    n_x = 11
    n_h = 256
    n_y = 3

    
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
        # np.convolve permet de lisser la courbe -> produit de convolution (intuitivement on fais glisser une fenêtre)
        moyenne_mobile = np.convolve(resultats, np.ones(window_size)/window_size, mode='valid') #moyenne arithmétique : np.ones(window_size)/window_size
                                                                                                #mode ='valid' = The convolution product is only given for points where the signals overlap completely
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
```

:<img width="1000" height="600" alt="progression_apprentissage_brut_in_grey_convolve_in_grey" src="https://github.com/user-attachments/assets/8be42dde-2990-4acd-add9-afddeb480d27" />
