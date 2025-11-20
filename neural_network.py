from layers import *
from data_generators import *
import matplotlib.pyplot as plt


class NeuralNetwork():
    """
    Neural network class that takes a list of layers
    and performs forward and backward pass, as well
    as gradient descent step.
    """

    def __init__(self,layers):
        #layers is a list where each element is of the Layer class
        self.layers = layers
    
    def forward(self,x):
        #Recursively perform forward pass from initial input x
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self,grad):
        """
        Recursively perform backward pass 
        from grad : derivative of the loss wrt 
        the final output from the forward pass.
        """

        #reversed yields the layers in reversed order
        for layer in reversed(self.layers):
            #print(layer)
            grad = layer.backward(grad)
        return grad
    
    def step_gd(self,alpha):
        """
        Perform a gradient descent step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            #Check if layer is of class a class that has parameters
            if isinstance(layer,(LinearLayer,EmbedPosition,FeedForward,Attention)):
                #print(layer)
                layer.step_gd(alpha)
        return
    
    def step_adam(self,alpha, j):
        """
        Perform a adam step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            #Check if layer is of class a class that has parameters
            if isinstance(layer,(LinearLayer,EmbedPosition,FeedForward,Attention)):
                #print(layer)
                layer.step_adam(alpha, j)
        return
    
    def steepest_descent_algoritmen(self, loss, X_batches, Y_batches, n_iter, tol=0.01):
        loss_arr = np.zeros((X_batches.shape[0], n_iter))
        j = 1
        L = 1
        while j <= n_iter and L > tol:
            for k in range(X_batches.shape[0]):
                #do a forward pass
                Z = self.forward(X_batches[k])

                #compute the loss
                L = loss.forward(Z, Y_batches[k])
                loss_arr[k, j-1] = L
                #get the derivative of the loss wrt Z
                grad_Z = loss.backward()
                #print(grad_Z.shape)
                #and perform a backward pass
                _ = self.backward(grad_Z)
                #print("her")
                #and and do a gradient descent step
                _ = self.step_gd(0.01)
            j += 1
        return loss_arr
    
    def adams_algoritmen(self, loss, X_batches, Y_batches, n_iter, tol):
        j = 1
        L = 1
        loss_arr = np.zeros((X_batches.shape[0], n_iter))
        while j <= n_iter and L > tol:
            for k in range(X_batches.shape[0]):
                #do a forward pass
                Z = self.forward(X_batches[k])

                #compute the loss
                L = loss.forward(Z, Y_batches[k])
                loss_arr[k, j-1] = L
                #get the derivative of the loss wrt Z
                grad_Z = loss.backward()
                #print(grad_Z.shape)
                #and perform a backward pass
                _ = self.backward(grad_Z)
                #print("her")
                #and and do a adam step
                _ = self.step_adam(0.01, j)
            j += 1
        return loss_arr
    
    def test(self, x_test, y_test, m):
        #initialiserer lokale variabler
        n_riktige = 0
        n_batches, b, n = x_test.shape   
        for i in range(n_batches):
            lokal_x = x_test[i]
            #genererer sortert liste
            for j in range(y_test.shape[2]):  
                X_test = onehot(lokal_x, m)
                Z = np.argmax(self.forward(X_test), axis=1)
                zeros = np.zeros((b, n + 1 + j))
                zeros[:, :n+j] = lokal_x
                lokal_x = zeros
                lokal_x[:, n+j] = Z[:,-1]
            #finner antall som er sortert riktig
            n_riktige += np.sum(np.where(np.all(y_test[i, :, :] == lokal_x[:, n:], axis=1), 1, 0))
        return n_riktige/(n_batches*b)
    


#######################################




        
        
    











#print(data["x_test"])
#print(data["y_test"])
#print(data["x_test"].shape)



#y_pred = additionNetwork.forward(X1)
#y_pred = np.argmax(y_pred, axis = 1)
#print(y_pred)
#y_act = onehot(y1)
#y_pred = y_pred[:]



'''xtest, ytest = data["x_test"], data["y_test"]
for i in range(xtest.shape[1]):
    Checked += 1
    x = xtest[i]
    X = onehot(x)
    additionNetwork.forward(x)
    x_res = np.argmax(X, axis = 1)
    x_res = x_res[]
    y_rev = xtest[i].reversed


'''






