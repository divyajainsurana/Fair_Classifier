# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sys
import time
import math
import data_adult as dp

start_time = time.time()
np.random.seed(1)
(X,Y),(xtest,ytest) = dp.load_data()
Y=Y[:,np.newaxis]

z=X[0:,9]
z_test = xtest[0:,9]
test_data=xtest
test_output = ytest
# Visualize the data:

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[0] # training set size
print (set(z))
"""
for i in range(len(X)):
    if(X[i][9]==-0.6927946734072801):
        X[i][9]=1
    else:
        X[i][9]=0
for i in range(len(test_data)):
    if(test_data[i][9]==1.4433812557800418):
        test_data[i][9]=0
    else:
        test_data[i][9]=1
print ('I have m = %d training examples!' % (m))
"""
def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s
def sigmoidPrime( s):
    #derivative of sigmoid
    return sigmoid(s) * (1.0 - sigmoid(s))
  
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = 14 # size of input layer
    n_h = 14
    n_y = 1# size of output layer
    
    return (n_x, n_h, n_y)
  
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1) 
    
    
    W1 = np.random.randn(14,14)
  #  b1 = np.zeros((n_x,1))*0.
    W2 = np.random.randn(14,14)
  #  b2 = np.zeros((n_h,1))
    W3 = np.random.randn(14,1)
    
    
    parameters = {"W1": W1,
                #  "b1": b1,
                  "W2": W2,
                  "W3": W3}
               #   "b2": b2}
    
    return parameters
  
  
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
  #  b1 = parameters["b1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
 #   b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(X,W1)#+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1,W2)#+b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2,W3)
    A3 = sigmoid(Z3)
   # assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}
    
    return A3, cache

def cov(x,y):

    if len(x) != len(y):
        return

    n = len(x)

    xy = [x[i]*y[i] for i in range(n)]

    mean_x = sum(x)/float(n)
    mean_y = sum(y)/float(n)

    return (sum(xy) - n*mean_x * mean_y) / float(n)
 

def compute_cost(A3, Y, parameters,z):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function, 
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook 
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    
    """
    
    m = Y.shape[0] # number of example

    # Compute the cross-entropy cost
    z=z[:,np.newaxis]
    covariance = cov(z,A3)
    logprobs = 0.6*(np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y))+0.4*(covariance)**2
    cost = 1./m * np.nansum(logprobs)
   
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
 #   assert(isinstance(cost, float))
    
    return cost
  

def backward_propagation(parameters, cache, X, Y,z):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[0]
   # print (m)
    # First, retrieve W1 and W2 from the dictionary "parameters".
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    # Retrieve also A1 and A2 from dictionary "cache".
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    z=z[:,np.newaxis]
    mat = np.full((31655,14),z)
    dC3 = (1/m*np.dot(np.transpose(A3),z))
    dC2 =(1/m*np.dot(np.transpose(A2),mat))
    #print (dC2.shape)
    dC1 =(1/m*np.dot(np.transpose(A1),mat))
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    covariance = cov(z,A2)
 #   print (covariance)
    #mat=np.full((24129,1),covariance)
    a=0.6
    dZ3 = (A3-Y)
    dW3 = 1/m*np.dot(np.transpose(A2),dZ3)
    dW3 = (0.6)*dW3 + 0.4*dC3
    dZ2 = np.dot(dZ3,np.transpose(W3))*sigmoidPrime(A3) 
    dW2 = 1/m*np.dot(np.transpose(A1),dZ2)
    #print (dW2.shape)
    dW2 = (0.6)*dW2+0.4*dC2
   # print(dW2.size, W2.size, A1.size,dZ2.size)
  #  db2 = 1/m * np.sum(dZ2,axis =1, keepdims =True)
    dZ1 = np.dot(dZ2,np.transpose(W2))*sigmoidPrime(A2)
    dW1 = 1/m * np.dot( np.transpose(X),dZ1)
    dW1 = 0.6*dW1+0.4*dC1
    grads = {"dW3": dW3,
             "dC3": dC3,
             "dW1": dW1,
             "dC2": dC2,
             "dW2": dW2,
             "dC1": dC1}
    
    return grads,covariance
  
  
 

def update_parameters(parameters, grads, covariance,learning_rate = 0.01):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
   
    W1 = parameters["W1"]
   # b1 = parameters["b1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
   # b2 = parameters["b2"]
    m =31655
    
    # Retrieve each gradient from the dictionary "grads"
   
    dW1 = grads["dW1"]
    dC1 = grads["dC1"]
    dW2 = grads["dW2"]
    dC2 = grads["dC2"]
    dW3 = grads["dW3"]
    dC3 = grads["dC3"]
    
    # Update rule for each parameter
    a=0.0
    W1 = W1 -learning_rate* dW1 #- a*covariance
  #  b1 = b1 - learning_rate*db1
    W2 = W2 -learning_rate* dW2 #-a*covariance
  #  b2 = b2 - learning_rate*db2
    
    W3 = W3 -learning_rate*dW3    
    parameters = {"W1": W1,
               #   "b1": b1,
                  "W2": W2,
                  "W3": W3}
                #  "b2": b2}
    
    return parameters
  
  
def nn_model(X, Y,z, n_h, num_iterations = 100, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters,z)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads,covariance = backward_propagation(parameters, cache, X, Y,z)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads,covariance)
        
        
        
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
  
  
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    # (≈ 2 lines of code)
    o, cache = forward_propagation(X,parameters)
    return o
  
  
  # Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y,z, n_h = 14, num_iterations = 10000, print_cost=True)

# Plot the decision boundary

predictions= predict(parameters, X)
for i in range(len(predictions)):
      if(predictions[i] > 0.50):
        predictions[i] =1
      else:
        predictions[i] =0
np.savetxt("data_new.npy",predictions,fmt="%s")
np.savetxt("label_new.npy",Y,fmt="%s")
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y, predictions)*100
print('Acc: {:.4f}'.format(acc))
out = predict(parameters, test_data)
count_m=0
count_f=0
prob_f=0
prob_m=0
prob_fpr_m=0
prob_fpr_f=0

for i in range(len(z_test)):
    if z_test[i]==0.6920970100140461:
        count_f+=1
    else:
        count_m+=1 
count =0
for i in range(len(out)):
      if(out[i] > 0.50):
        out[i] =1
       # count+=1
      else:
        out[i] =0

      
for i,j,k in zip(out,z_test,test_output):
    if j==0.6920970100140461 and i ==1:
        prob_f+=1
      #  print(j,i)
    elif j==-1.4448841499542169 and i==1:
        prob_m+=1

prob_f_dm,prob_m_dm,tp_s1,tp_s2,tn_s1,tn_s2=0,0,0,0,0,0

for i,j,k in zip(out,z_test,test_output):
    if j==0.6920970100140461 and i ==1 and k==0:
        prob_f_dm+=1
      #  print(j,i)
    elif j==0.6920970100140461 and i==1 and k==1:
        tp_s1+=1
    elif j==0.6920970100140461 and i==0 and k==0:
         tn_s1+=1
    elif j==-1.4448841499542169 and i==1 and k==0:
        prob_m_dm+=1
prob_f_dm_n,prob_m_dm_n=0,0
for i,j,k in zip(out,z_test,test_output):
    if j==0.6920970100140461 and i ==0 and k==1:
        prob_f_dm_n+=1
      #  print(j,i)
    elif j==-1.4448841499542169 and i==0 and k==1:
        prob_m_dm_n+=1  
    elif j==-1.4448841499542169 and i==1 and k==1:
        tp_s2+=1
    elif j==-1.4448841499542169 and i==0 and k==0:
         tn_s2+=1

fpr_s1 = (prob_f_dm)/(prob_f_dm+tn_s1)
fpr_s2 = (prob_f_dm_n)/(prob_f_dm_n+tn_s2)
print ("FPRs1,FPRs2,FPR",fpr_s1,fpr_s2,fpr_s1-fpr_s2)
fnr_s1 = (prob_f_dm_n)/(prob_f_dm_n+tp_s1)
fnr_s2 = (prob_m_dm_n)/(prob_m_dm_n+tp_s2)
print ("FNRs1,FNRs2,FNR",fnr_s1,fnr_s2,fnr_s1-fnr_s2)
DI_n,DI_d= (prob_f/count_f),(prob_m/count_m)
MT_n,MI_d = (prob_f),(prob_m)
print (((DI_n,DI_d)))
print ("Disparate Imapct",abs(DI_n-DI_d))
print ("p% rule",min(DI_n/DI_d,DI_d/DI_n))
accu = accuracy_score(test_output, out)*100
print('Test Acc: {:.4f}'.format(accu))
cova = cov(z_test,out)
print ("covarianve",cova)

# This may take about 2 minutes to run
print("--- %s seconds ---" % (time.time() - start_time))

