import numpy as np 
import os
import matplotlib.pyplot as plt

#grille de jeu 10x10 --- -1 apple --- 0 blank --- 1 tail --- tail stocké dans une liste --- Head en dernière position (car np.append)
#                                                                                       --- Tail en première position
#constantes influençant le taux d'apprentissage -- alpha et gamma
#3 possibilités pour l'action a (tout droit, droite, gauche)
#a=0=tout droit -- a=1=droite -- a=2=gauche
#3 possibilités pour la direction du danger immédiat d (tout droit, droite, gauche)
#d=0=tout droit -- d=1=droite -- d=2=gauche

#4 possibilités pour les positions/directions stockés dans l'état X (vecteur à 11 paramètres)
#0=droite -- 1=haut -- 2=gauche -- 3=bas

gamma = 0.9
alpha = 0.001

def forward(X, W1, B1, W2, B2):
    X = X.reshape(-1, 1) 
    
    Z1 = np.dot(W1, X) + B1
    A1 = np.maximum(0, Z1) # Version vectorisée de ReLU

    Z2 = np.dot(W2, A1) + B2
    Q_pred = Z2.reshape(-1, 1)

    return Q_pred, Z1, A1, Z2


def decision(epsilon, Q):
    random_value = np.random.random()
    if random_value < epsilon :
        random_value2 = np.random.choice(np.array([0,1,2]))
        return random_value2
    else :
        return np.argmax(Q)


def action(jeu,X,a,count_apples):
    #etat du jeu
    grille = jeu[0]
    tail = jeu [1]
    head_i_old,head_j_old = tail[len(tail)-1]

    #variables à calculer
    r=0
    game_over = False


    ##traiter tous les cas de mouvements de la tête et update X direction tête
    arg_dir = np.argmax(np.array([X[3],X[4],X[5],X[6]]))
    X[3]=0
    X[4]=0
    X[5]=0
    X[6]=0
    #variables intermédiaires à calculer
    danger=False
    (i,j)=(0,0)

    if a==0:
        if X[0]:
            danger = True
        if arg_dir==0:
            (i,j) = (0,1)
            X[3]=1
        elif arg_dir==1:
            (i,j) = (-1,0)
            X[4]=1
        elif arg_dir==2:
            (i,j)=(0,-1)
            X[5]=1
        elif arg_dir==3:
            (i,j) = (1,0)
            X[6]=1
    elif a==1:
        if X[1]:
            danger=True
        if arg_dir==0:
            (i,j) = (1,0)
            X[6]=1
        elif arg_dir==1:
            (i,j) = (0,1)
            X[3]=1
        elif arg_dir==2:
            (i,j)=(-1,0)
            X[4]=1
        elif arg_dir==3:
            (i,j) = (0,-1)
            X[5]=1
    if a==2:
        if X[2]:
            danger = True
        if arg_dir==0:
            (i,j) = (-1,0)
            X[4]=1
        elif arg_dir==1:
            (i,j) = (0,-1)
            X[5]=1
        elif arg_dir==2:
            (i,j)=(1,0)
            X[6]=1
        elif arg_dir==3:
            (i,j) = (0,1)
            X[3]=1


    ##update tail,update grille,find game_over,find_r
    if danger:
        game_over=True
        r=-10
    else:
        (head_i_new,head_j_new) = (head_i_old+i,head_j_old+j)
        tail.append((head_i_new,head_j_new))
        r=-0.2
        #2 cases -> eat an apple -> add new head
        #        -> don't eat any apple -> remove tail & add new head  

        if grille[head_i_new,head_j_new]==-1:
            grille[head_i_new,head_j_new] = 1
            r=10 
            count_apples+=1
            # Trouve les indices où la grille est vide (0)
            empty_cells = np.argwhere(grille == 0) 
            # Choisi un index au hasard parmi les cellules vides
            idx = np.random.choice(len(empty_cells))
            apple_i_new, apple_j_new = empty_cells[idx]

            grille[apple_i_new,apple_j_new] = -1
            #update pos_apple

            X[7]=0
            X[8]=0
            X[9]=0
            X[10]=0

              
            if head_j_new<=apple_j_new:
                X[7]=1
            else:
                X[9]=1
            if head_i_new<=apple_i_new:
                X[10]=1
            else:
                X[8]=1


        else:
            grille[head_i_new,head_j_new] = 1
            tail_i_old,tail_j_old = tail[0] 
            tail.pop(0)
            grille[tail_i_old,tail_j_old]=0


        #update danger
        X[0]=0
        X[1]=0
        X[2]=0
        

        def is_danger(check_i, check_j):
            if check_i < 0 or check_i >= 10 or check_j < 0 or check_j >= 10 or grille[check_i,check_j] == 1:
                return 1 
            else:
                return 0        

        if X[3]:
            X[0]=is_danger(head_i_new,head_j_new+1)
            X[1]=is_danger(head_i_new+1,head_j_new)
            X[2]=is_danger(head_i_new-1,head_j_new)
        elif X[4]:
            X[0]=is_danger(head_i_new-1,head_j_new)
            X[1]=is_danger(head_i_new,head_j_new+1)
            X[2]=is_danger(head_i_new,head_j_new-1)
        elif X[5]:
            X[0]=is_danger(head_i_new,head_j_new-1)
            X[1]=is_danger(head_i_new-1,head_j_new)
            X[2]=is_danger(head_i_new+1,head_j_new)
        elif X[6]:
            X[0]=is_danger(head_i_new+1,head_j_new)
            X[1]=is_danger(head_i_new,head_j_new-1)
            X[2]=is_danger(head_i_new,head_j_new+1)

    
    return X,r,game_over,count_apples
    
def sigma_D(Z):
    return (Z > 0).astype(float) # Version vectorisée

def backward(X, Q_pred, Q_target, W1, B1, W2, B2, Z1, A1, Z2):
    X = X.reshape(-1, 1)
    
    # Delta Out (3, 1)
    delta_out = Q_pred- Q_target
    
    # Gradients Layer 2
    DL_DW2 = np.dot(delta_out, A1.T) # (3, 1) . (1, 256) -> (3, 256)
    DL_DB2 = delta_out
    
    # Gradients Layer 1
    # Terme d'erreur rétropropagé
    delta_hidden = np.dot(W2.T, delta_out) * sigma_D(Z1) # Produit matriciel puis produit element-wise
    
    DL_DW1 = np.dot(delta_hidden, X.T) # (256, 1) . (1, 11) -> (256, 11)
    DL_DB1 = delta_hidden
    
    # Update
    W1 = W1 - alpha * DL_DW1
    W2 = W2 - alpha * DL_DW2
    B1 = B1 - alpha * DL_DB1
    B2 = B2 - alpha * DL_DB2
    
    return W1, B1, W2, B2


##training réalise l'entièreté d'une séance d'entraînement --> boucle for pour répéter la séance d'entraînement (opti?)


def training_session(epsilon,W1,B1,W2,B2):
    game_over = False
    count_apples=0
    grille = np.zeros((10,10))
    grille[5,3] = 1 
    grille[5,2] = 1
    grille[5,6] = -1
    X = np.array([0,0,0, 1,0,0,0, 1,0,0,0])
    tail = [(5,2),(5,3)]
    jeu = (grille,tail)

    while not game_over:
        Q_pred,Z1,A1,Z2=forward(X,W1,B1,W2,B2)
        a=decision(epsilon,Q_pred)
        X_old = X.copy() # On sauvegarde l'état actuel pour le backward
        X_new, r, game_over,count_apples = action(jeu, X, a,count_apples) # X est modifié ici, X_new pointe dessus

        Q_target = np.copy(Q_pred)
        if game_over:
            Q_target[a, 0] = r
        else:
            Q_tmp,_,_,_=forward(X_new,W1,B1,W2,B2)
            Q_target[a, 0] =r+gamma*np.max(Q_tmp)
        
        
        W1, B1, W2, B2 = backward(X_old, Q_pred, Q_target, W1, B1, W2, B2, Z1, A1, Z2)
        X=X_new


    return W1,B1,W2,B2,count_apples

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
        # Sinon, on initialise tout à neuf 
        W1 = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)
        B1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * np.sqrt(2/n_h)
        B2 = np.zeros((n_y, 1))
        epsilon = 1.0 # On commence à 100% exploration
        nb_iterations=0
    
    
    # Test
    resultats = []

    print("Démarrage training...")
    iterations=100
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



main()











