import numpy as np 
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



def main():
    # Exemple d'initialisation pour tester les dimensions
    n_x = 11
    n_h = 256
    n_y = 3
    
    W1 = np.random.rand(n_h, n_x) * 0.1
    B1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.1
    B2 = np.zeros((n_y, 1))
    
    
    # Test
    epsilon=1
    print("Démarrage training...")

    for i in range(10000):
        W1, B1, W2, B2,c = training_session(epsilon, W1, B1, W2, B2)
        epsilon=max(0.01,epsilon*0.999)
        print(epsilon,c)
    print("Training terminé sans erreur !")


main()
"""
#remarques pour upgrade
#Pièges de programmation! (learning notes)
ceci ne mets à jour x dans la fonction locale update !!!
def test():
    x=7
    def update(x):
        x+=1
    print(x)

ceci renvoie Hello car le or s'arrête dès qu'il trouve un True !!
x=[]
if 1==1 or x[5]:
    print("Hello")


ceci copie un tableau mais attention
tab=np.array([1])
tab2=np.copy(tab)
tab2[0]=15
print("{} et {}".format(tab[0],tab2[0])) -> 1 et 15
ceci ne fonctionnera pas !!
tab=np.array([1])
tab2=tab
tab2[0]=15
print("{} et {}".format(tab[0],tab2[0])) -> 15 et 15
"""