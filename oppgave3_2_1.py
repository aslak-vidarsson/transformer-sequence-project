from layers import *
from data_generators import *
import matplotlib.pyplot as plt
from neural_network import *

def network_maker(d, k, p, m, n_max):
    """
    Konstruerer og returnerer et nettverk
    """

    #Lager nødvendige lag
    feed_forward = FeedForward(d,p)
    attention = Attention(d,k)
    feed_forward2 = FeedForward(d,p)
    attention2 = Attention(d,k)
    embed_pos = EmbedPosition(n_max,m,d)
    un_embed = LinearLayer(d,m)
    softmax = Softmax()

    #initialiserer nettverket ved hjelp av lagene initialisert over
    network = NeuralNetwork([embed_pos,  attention, feed_forward,  attention2, feed_forward2, un_embed, softmax])
    
    #initialiserer lossfunksjon
    loss = CrossEntropy()

    #returnerer nettverk og lossfunksjon
    return network, loss

def oppgave3_2(m, n_iter, network, train_network, loss, label, train_data, test_data):
    """
    Utfører trening og testing av et nettverk, på trenings og test-data
    """

    #Henter trenings- og test-data
    x_batches = train_data["x_train"]
    y_batches = train_data["y_train"]
    x_test = test_data["x_test"]
    y_test = test_data["y_test"]


    #Kjører onehot på hver batch
    X_batches = batchmaker(x_batches, m)
    Y_batches = batchmaker(y_batches, m)

    #Tester hvor stor andel blir sorert korrekt før trening
    print("Andel korrekt sortert før trening med "+label, network.test(x_test, y_test, m))
    #Trener nettverket
    loss_arr = train_network(loss, X_batches, Y_batches, n_iter, 10**-2)
    #Tester hvor stor andel blir sorert korrekt etter trening
    print("Andel korrekt sortert etter trening med "+label, network.test(x_test, y_test, m))
    #Plotter loss-funksjonen som funksjon av trening
    plt.semilogy(np.sum(loss_arr, axis=0)/loss_arr.shape[0], label=label)
    


def super_oppgave_3_2(length, m, d, k ,p, n_iter, samples_per_batch=250, n_batches_train = 10, n_batches_test = 10):
    #Henter test og treningsdata fra get_train_test_sorting
    train_data = get_train_test_sorting(length=length, num_ints=m,samples_per_batch=samples_per_batch,n_batches_train=n_batches_train, n_batches_test=n_batches_test)
    test_data = get_train_test_sorting(length=length, num_ints=m,samples_per_batch=samples_per_batch,n_batches_train=n_batches_train, n_batches_test=n_batches_test)

    #Lager et nettverk og loss-funksjonen
    network, loss = network_maker(d=d, k=k, p=p, m=m, n_max = length*2-1)
    #Bruker adams-algoritmen til å trene nettverket
    oppgave3_2(m=m, n_iter = n_iter, network=network, train_network=network.adams_algoritmen, loss=loss, label="Step adam", train_data=train_data, test_data=test_data)
    
    #Lager et nytt nettverk og loss-funksjon
    network, loss = network_maker(d=d, k=k, p=p, m=m, n_max = length*2-1)
    #Bruker steepest-descent-algoritmen til å trene nettverket
    oppgave3_2(m=m, n_iter = n_iter, network=network, train_network=network.steepest_descent_algoritmen, loss=loss, label="Steepest descent", train_data=train_data, test_data=test_data)
    
    #Gir navn på aksene og viser plottet med begge grafer
    plt.xlabel("Iterasjoner")
    plt.ylabel("Lossfunksjon")
    plt.legend()
    plt.show()
