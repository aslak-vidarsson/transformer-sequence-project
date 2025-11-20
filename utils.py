import numpy as np


def onehot(x,m):
    """
    Input:
    - x : np.array of integers with shape (b,n)
             b is the batch size and 
             n is the number of elements in the sequence
    - m : integer, number of elements in the vocabulary 
                such that x[i,j] <= m-1 for all i,j

    Output:     
    - x_one_hot : np.array of one-hot encoded integers with shape (b,m,n)

                    x[i,j,k] = 1 if x[i,j] = k, else 0 
                    for all i,j
    """

    b,n = x.shape

    #Making sure that x is an array of integers
    x = x.astype(int)
    x_one_hot = np.zeros((b,m,n))
    x_one_hot[np.arange(b)[:,None],x,np.arange(n)[None,:]] = 1
    return x_one_hot


#Laget en funksjon som gjør en matrise med dimensjon (antall batches, datapunkter per batch, lengden til datapunktet) 
#til (antall batches, datapunkter per batch, mulige, lengden til datapunktet) altså onehot-representasjon
#Denne egner seg å bruke med adam og SD algoritmene, som eksempelvis vist i oppgave3_4.py
def batchmaker(x_batches, m):
    n_batches, b, n_max = x_batches.shape
    X_batches = np.zeros((n_batches, b, m, n_max))
    for i in range(n_batches):
        X_batches[i] = onehot(x_batches[i], m)
    return X_batches
