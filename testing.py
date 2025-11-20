from layers import *
from data_generators import *
import matplotlib.pyplot as plt
from neural_network import *
#We choose some arbitrary values for the dimensions
b = 13
n_max = 12
m = 9
n = 6

d = 15
k = 14
p = 21


#Create an arbitrary dataset
x = np.random.randint(0, m, (b,n))
y = np.random.randint(0, m, (b,n_max))

#initialize the layers
feed_forward = FeedForward(d,p)
attention = Attention(d,k)
embed_pos = EmbedPosition(n_max,m,d)
un_embed = LinearLayer(d,m)
softmax = Softmax()




#a manual forward pass
X = onehot(x, m)
z0 = embed_pos.forward(X)
z1 = feed_forward.forward(z0)
z2 = attention.forward(z1)
z3 = un_embed.forward(z2)
Z = softmax.forward(z3)






#check the shapes
assert X.shape == (b,m,n), f"X.shape={X.shape}, expected {(b,m,n)}"
assert z0.shape == (b,d,n), f"z0.shape={z0.shape}, expected {(b,d,n)}"
assert z1.shape == (b,d,n), f"z1.shape={z1.shape}, expected {(b,d,n)}"
assert z2.shape == (b,d,n), f"z2.shape={z2.shape}, expected {(b,d,n)}"
assert z3.shape == (b,m,n), f"z3.shape={z3.shape}, expected {(b,m,n)}"
assert Z.shape == (b,m,n), f"Z.shape={Z.shape}, expected {(b,m,n)}"

#is X one-hot?
assert X.sum() == b*n, f"X.sum()={X.sum()}, expected {b*n}"


assert np.allclose(Z.sum(axis=1), 1), f"Z.sum(axis=1)={Z.sum(axis=1)}, expected {np.ones(b)}"
assert np.abs(Z.sum() - b*n) < 1e-5, f"Z.sum()={Z.sum()}, expected {b*n}"
assert np.all(Z>=0), f"Z={Z}, expected all entries to be non-negative"


#test the forward pass
x = np.random.randint(0, m, (b,n_max))
X = onehot(x, m)

#we test with a y that is shorter than the maximum length
n_y = n_max -1 
y = np.random.randint(0, m, (b,n_y))

#initialize a neural network based on the layers above
network = NeuralNetwork([embed_pos, feed_forward, attention, un_embed, softmax])
#and a loss function
loss = CrossEntropy()

#do a forward pass
Z = network.forward(X)

#compute the loss
L = loss.forward(Z, y)
print("L", L)
#get the derivative of the loss wrt Z
grad_Z = loss.backward()
#print(grad_Z.shape)
#and perform a backward pass
_ = network.backward(grad_Z)
#print("her")
#and and do a gradient descent step
_ = network.step_gd(0.01)

#do a forward pass
Z = network.forward(X)

#compute the loss
L = loss.forward(Z, y)
print("L2", L)




network.steepest_descent_algoritmen(loss, X, y, 20)
Z = network.forward(X)

#compute the loss
L = loss.forward(Z, y)
print("L3", L)


network.adams_algoritmen(loss, np.array([X]), np.array([y]), 20, 10**-8)

Z = network.forward(X)

#compute the loss
L = loss.forward(Z, y)
print("L4", L)

m = 2
n_batches = 10
data = get_train_test_sorting(length=5, num_ints=m,samples_per_batch=250,n_batches_train=n_batches, n_batches_test=2)
data2 = get_train_test_sorting(length=5, num_ints=m,samples_per_batch=250,n_batches_train=n_batches, n_batches_test=2)

x_batches = data["x_train"]
y_batches = data["y_train"]
x_test = data2["x_test"]
y_test = data2["y_test"]
print("x.shape",x.shape)
b = x_batches.shape[1]
n_max = x_batches.shape[2]

n = y_batches.shape[2]

d = 10
k = 5
p = 15



X_batches = np.zeros((n_batches, b, m, n_max))
for i in range(n_batches):
    X_batches[i] = onehot(x_batches[i], m)

feed_forward = FeedForward(d,p)
attention = Attention(d,k)
embed_pos = EmbedPosition(n_max,m,d)
un_embed = LinearLayer(d,m)
softmax = Softmax()

#initialize a neural network based on the layers above
network = NeuralNetwork([embed_pos,  feed_forward, feed_forward, attention, un_embed, softmax])
#and a loss function
loss = CrossEntropy()
#network.adams_algoritmen(loss, X, y, 100, 10**-8)

"""
print("Andel korrekte sorteringer uten trening", network.test(x_test, y_test, m))
loss_arr = np.zeros((X_batches.shape[0], 0))
trainings = 100
success_arr = np.zeros(trainings)
for p in range(trainings):
    loss_arr = np.concatenate((loss_arr, network.adams_algoritmen(loss, X_batches, y_batches, 1, 10**-2)), axis=1)
    test_data = network.test(x_test, y_test, m)
    success_arr[p] = test_data
    #print("Andel korrekte sorteringer etter "+str(p+1)+" runder trening", test_data)
"""
"""
steg1, steg2 = 10, 20 
loss_arr = network.adams_algoritmen(loss, X_batches, y_batches, steg1, 10**-2)
print("Andel korrekte sorteringer etter "+str(steg1)+" runder trening", network.test(x_test, y_test, m))
loss_arr = np.concatenate(loss_arr, network.adams_algoritmen(loss, X_batches, y_batches, steg2, 10**-2))
print("Andel korrekte sorteringer etter "+str(steg1+steg2)+" runder trening", network.test(x_test, y_test, m))
"""
loss_arr = network.adams_algoritmen(loss, X_batches, y_batches, 150, 10**-2)
print("Andel korrekte sorteringer", network.test(x_test, y_test, m))
plt.semilogy(np.sum(loss_arr, axis=0)/loss_arr.shape[0])
plt.show()
#plt.plot(success_arr)
#plt.show()


x = np.array([[1, 0,1, 1, 0]])
print("x(0)", x)
for i in range(5):
    X = onehot(x, 2)
    Z = np.argmax(network.forward(X), axis=1)
    zeros = np.zeros((1,6+i))
    zeros[:, :-1] = x
    x = zeros
    x[:, -1] = Z[:, -1]
    print("x("+str(i+1)+")",x)

