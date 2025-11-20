import numpy as np
from utils import *


class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']

    #Implementasjon av step adam
    def step_adam(self, alpha, j):

        #Variabler
        b1 = 0.9
        b2 = 0.999
        epsilon = 10**(-8)

        #Selve steget
        for param in self.params:
            G  = self.params[param]['d']
            self.params[param]['M'] = b1 * self.params[param]['M'] + (1-b1) * G
            self.params[param]['V'] = b2 * self.params[param]['V'] + (1-b2) * (G * G)
            M_hat = (1/(1-b1**j)) * self.params[param]['M']
            V_hat = (1/(1-b2**j)) * self.params[param]['V']
            self.params[param]['w'] -= alpha*(M_hat / (np.sqrt(V_hat) + epsilon))



            
    


#Implementerer attention-laget
class Attention(Layer):

    def __init__(self,d, k):
        
        #Definerer de lineære lagene. De transponerte matrisene i forward endrer vi posisjon på d og k 
        self.linear_wk = LinearLayer(d, k)
        self.linear_wq_trans = LinearLayer(k, d)
        self.linear_wv = LinearLayer(d, k)
        self.linear_wo_trans = LinearLayer(k, d)

        #Definerer softmax-laget
        self.softmax_layer = Softmax()
        

    def forward(self,z):

       #lager D-matrise
        b = z.shape[0]
        n = z.shape[2]
        D = np.zeros((b, n, n))
        i1,i2 = np.tril_indices(n,-1)
        D[:,i1,i2] -= np.inf

        #lagrer z slik at den kan brukes i backward
        self.z = z

        #finner så forward substitution
        z_l_step1 = self.linear_wk.forward(z)
        z_l_step2 = self.linear_wq_trans.forward(z_l_step1)
        z_l_step3 = np.einsum("bnd, bdm -> bnm", z.transpose(0, 2, 1), z_l_step2, optimize=True)
        z_l_step4 = z_l_step3 + D

        #lagrer A slik at den brukes i backward
        self.A = self.softmax_layer.forward(z_l_step4)

        #Fortsetter på forward substitution
        z_l_step5 = np.einsum("bdm, bmn -> bdn", z, self.A, optimize=True)
        z_l_step6 = self.linear_wv.forward(z_l_step5)
        z_l_step7 = self.linear_wo_trans.forward(z_l_step6)
        z_l = z_l_step7 + z

        return z_l


    def backward(self,grad):
        #Matriser som brukes senere
        g_OV = self.linear_wv.backward((self.linear_wo_trans.backward(grad)))
        g_s = self.softmax_layer.backward(np.einsum("bdn, bnm -> bdm", self.z.transpose(0, 2, 1), g_OV, optimize=True)) 

        #Alle leddene som inngår i backward substitution
        ret1 = grad
        ret2 = np.einsum("bnd, bdc -> bnc", g_OV, self.A.transpose(0,2,1), optimize=True)
        ret3 = self.linear_wk.backward(self.linear_wq_trans.backward(np.einsum("bnd, bdm -> bnm", self.z, g_s, optimize=True)))
        ret4 = self.linear_wq_trans.forward(self.linear_wk.forward(np.einsum("bnd, bdm -> bnm", self.z, g_s.transpose(0, 2, 1), optimize=True)))

        return ret1 + ret2 +ret3 + ret4
    
    #Oppdaterer parameterer i de lineære lagene med gradient descent
    def step_gd(self, alpha):
        self.linear_wk.step_gd(alpha)
        self.linear_wo_trans.step_gd(alpha)
        self.linear_wq_trans.step_gd(alpha)
        self.linear_wv.step_gd(alpha)
    
    #Oppdaterer paramanterer i de lineære lagene med step adam
    def step_adam(self, alpha, j):
        self.linear_wk.step_adam(alpha, j)
        self.linear_wo_trans.step_adam(alpha, j)
        self.linear_wq_trans.step_adam(alpha, j)
        self.linear_wv.step_adam(alpha, j)
    

#implementerer softmax-alget
class Softmax(Layer):

    def __init__(self):
        #epsilon gjør at man unngår divisjon på null
        self.epsilon = 10**-8
        return

    #implementerer forward-steget for softmax
    def forward(self,x):
       
        #Regner ut
        P = np.exp(x - x.max(axis=1,keepdims=True))
        Q = np.sum(P,axis=1,keepdims=True)
        z_l = P/(Q+self.epsilon)
        self.P = P
        self.Q = Q
        self.z_l = z_l

        return z_l


    #implementerer backward-steget for softmax
    def backward(self,grad):
        #S er en hjelpevariabel
        S = self.P/(self.Q*self.Q+self.epsilon)
        g_l1 = grad*self.z_l - np.sum(grad*S,axis=1,keepdims=True)*self.P
        return g_l1


#Implementerte crossentropy-funksjonen - funksjonen som skal minimeres
class CrossEntropy(Layer):

    def __init__(self):
        #definerer epsilon
        self.epsilon = 10**-8
        return

        
    #Forward-steg for crossentropy
    def forward(self,Y_h, Y):
        #Lagrer y_h som kommer fra det nevrale nettverket
        self.Y_h = Y_h
        
        #Slicer y_h til å samme lengde som den faktiske y
        Y_h = Y_h[:, :, Y_h.shape[2]-Y.shape[2]:]

        #Regner så verdien
        b, m, n = Y_h.shape
        self.Y = Y
        one_matrix = np.ones((b, 1, m))
        p = np.einsum("brm, bmn->brn", one_matrix, Y_h*Y, optimize=True)
        q = -np.log(p)
        l = np.sum(q)/(b*n)
        return l

    #backward-steget
    def backward(self, max_value=1):
        #Lager en nullmatrise 
        Y = np.zeros(self.Y_h.shape)

        #legger inn verdier i Y
        Y[:, :, self.Y_h.shape[2]-self.Y.shape[2]:] = self.Y
        g_l1 = -1/Y.shape[2]*(Y/(self.Y_h+self.epsilon))
        return g_l1
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w), 'V':0, 'M':0}} #Definerte V og M som 0 i starten for å bruke til adamsteget
        
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """
        #print("in forward")
        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x, optimize=True)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """
        #print("in backward")
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x, optimize=True)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad, optimize=True)
    

    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None, 'V':0, 'M':0}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)

    def step_adam(self,step_size, j):

        #We need to call the step_gd method of the linear layer
        self.embed.step_adam(step_size, j)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_adam(step_size, j)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)

    def step_adam(self,step_size, j):

        #Call the step_gd method of the linear layers
        self.l1.step_adam(step_size, j)
        self.l2.step_adam(step_size, j)


##########################################################
        



