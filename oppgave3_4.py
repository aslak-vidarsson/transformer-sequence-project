from layers import *
from data_generators import *
from neural_network import *


#Denne funksjonen printer ut hvor mange riktige tall det nevrale nettverket klarer å addere blant de 10_000 mulighetene med to tosidrede tall som kan legges sammen.
def oppgave3_4():
    #variabler for addisjonen
    n_digits = 2
    n = 3*n_digits
    m = 10

    #variabler for treningen og testingen
    samples_per_batch = 250
    n_batches_train = 20
    n_batches_test = 20
    #Variabler for lagene
    d = 30
    k = 20
    p = 40

    #Implementasjon av lagene (L = 3) for adam_nettverket
    adam_embed_position = EmbedPosition(n, m, d)

    adam_ff1 = FeedForward(d, p)
    adam_at1 = Attention(d, k)

    adam_ff2 = FeedForward(d, p)
    adam_at2 = Attention(d, k)

    adam_ff3 = FeedForward(d, p)
    adam_at3 = Attention(d, k)

    adam_un_embed = LinearLayer(d, m)
    adam_softmax = Softmax()

    adam_Loss = CrossEntropy()

    #Implementasjon av lagene (L = 3) for sd_nettverket
    sd_embed_position = EmbedPosition(n, m, d)

    sd_ff1 = FeedForward(d, p)
    sd_at1 = Attention(d, k)

    sd_ff2 = FeedForward(d, p)
    sd_at2 = Attention(d, k)

    sd_ff3 = FeedForward(d, p)
    sd_at3 = Attention(d, k)

    sd_un_embed = LinearLayer(d, m)
    sd_softmax = Softmax()

    sd_Loss = CrossEntropy()

    #nettverkene
    adam_additionNetwork = NeuralNetwork([adam_embed_position, adam_at1, adam_ff1, adam_at2, adam_ff2, adam_at3, adam_ff3, adam_un_embed, adam_softmax])
    sd_additionNetwork = NeuralNetwork([sd_embed_position, sd_at1, sd_ff1, sd_at2, sd_ff2, sd_at3, sd_ff3, sd_un_embed, sd_softmax])


    #trener nettverkene
    data = get_train_test_addition(n_digits,samples_per_batch,n_batches_train, n_batches_test)
    x, y = data["x_train"], data["y_train"]

    X_batches = batchmaker(x, m)
    Y_batches = batchmaker(y[:,:,-3:], m)

    loss_arr_adam = adam_additionNetwork.adams_algoritmen(adam_Loss, X_batches, Y_batches, 150, tol = 10**-2)
    loss_arr_sd = sd_additionNetwork.steepest_descent_algoritmen(sd_Loss, X_batches, Y_batches, 150, tol = 10**-2)

    #Antall korrekte
    correct_adam_test = 0
    correct_adam_train = 0
    correct_sd_test = 0
    correct_sd_train = 0

    #Itererer over batchene i testdataen
    for nr_batch in range(n_batches_test):
        x_adam = data["x_test"][nr_batch]
        x_sd = data["x_test"][nr_batch]
        y = data["y_test"][nr_batch]
        for _ in range(3):
            X_adam = onehot(x_adam, m)
            Z_adam = adam_additionNetwork.forward(X_adam)
            x_adam_pred = np.argmax(Z_adam, axis = 1)

            X_sd = onehot(x_sd, m)
            Z_sd = sd_additionNetwork.forward(X_sd)
            x_sd_pred = np.argmax(Z_sd, axis = 1) 
        
            new_x_adam = np.zeros((x_adam.shape[0], x_adam.shape[1] +1))

            new_x_adam[::,:new_x_adam.shape[1] -1] = x_adam[::, ::]
            new_x_adam[::, -1:] = x_adam_pred[::, -1:]
            x_adam = new_x_adam

            new_x_sd = np.zeros((x_sd.shape[0], x_sd.shape[1] +1))

            new_x_sd[::,:new_x_sd.shape[1] -1] = x_sd[::, ::]
            new_x_sd[::, -1:] = x_sd_pred[::, -1:]
            x_sd = new_x_sd
        y_pred_adam = x_adam[::, -3::]
        y_pred_sd = x_sd[::, -3::]
        y_faktisk = y[::, ::]
        for k in range(samples_per_batch):
            if (y_pred_adam[k][::-1] == y_faktisk[k]).all():
                correct_adam_test += 1

            if (y_pred_sd[k][::-1] == y_faktisk[k]).all():
                correct_sd_test += 1
    
    #Itererer over batchene i treningsdataen
    for nr_test_batches in range(n_batches_train):
        x_adam = data["x_train"][nr_test_batches]
        x_sd = data["x_train"][nr_test_batches]
        y = data["y_train"][nr_test_batches]
        x_adam = x_adam[:, :4:] 
        x_sd = x_sd[:, :4:]   
        for _ in range(3):
            X_adam = onehot(x_adam, m)
            X_sd = onehot(x_sd, m)
            Z_adam = adam_additionNetwork.forward(X_adam)
            Z_sd = sd_additionNetwork.forward(X_sd)
            x_pred_adam = np.argmax(Z_adam, axis = 1)
            x_pred_sd = np.argmax(Z_sd, axis = 1)
    
            new_x_adam = np.zeros((x_adam.shape[0], x_adam.shape[1] +1))
            new_x_sd = np.zeros((x_sd.shape[0], x_sd.shape[1] +1))


            new_x_adam[::,:new_x_adam.shape[1] -1] = x_adam[::, ::]
            new_x_adam[::, -1:] = x_pred_adam[::, -1:]
            x_adam = new_x_adam

            new_x_sd[::,:new_x_sd.shape[1] -1] = x_sd[::, ::]
            new_x_sd[::, -1:] = x_pred_sd[::, -1:]
            x_sd = new_x_sd
        
        y_pred_adam = x_adam[:, -3:]
        y_pred_sd = x_sd[:, -3:]
        y_faktisk = y[:, -3:]
        for k in range(samples_per_batch):
            if (y_pred_adam[k][::] == y_faktisk[k]).all():
                correct_adam_train += 1
            if (y_pred_sd[k][::] == y_faktisk[k]).all():
                correct_sd_train += 1

    #Printer resultater
    print("korrekt test Adam: ", correct_adam_test)
    print("korrekt train Adam: ", correct_adam_train)
    print("Prosentandel riktig Adam: ", (correct_adam_test + correct_adam_train)/10_000)
    print("korrekt test SD: ", correct_sd_test)
    print("korrekt train SD: ", correct_sd_train)
    print("Prosentandel riktig SD: ", (correct_sd_test + correct_sd_train)/10_000)
    
    #Returnerer gjennomsnittlig loss per batch
    avg_loss_adam = np.sum(loss_arr_adam, axis = 0)/loss_arr_adam.shape[0]
    avg_loss_sd = np.sum(loss_arr_sd, axis = 0)/loss_arr_sd.shape[0]

    return avg_loss_adam, avg_loss_sd
