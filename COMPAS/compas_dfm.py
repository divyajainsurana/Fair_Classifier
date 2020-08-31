# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sys
import time
import math
import data_preprocessed as dp
from scipy.special import expit
import numpy
start_time = time.time()
np.random.seed(1)
(X,Y),(xtest,ytest) = dp.load_data()
Y=Y[:,np.newaxis]
#print (X, Y.shape,xtest.shape)
m = Y.shape[0]
J1=[]
J2 =[]
for i in range(m):
            if (X[i][0]==0.4929843922796267):
                J1.append(X[i])
            else:
                J2.append(X[i])
J1 = np.array(J1)
J2 = np.array(J2)
#X =(np.vstack((J1,J2)))
z_test1 = xtest[0:,3]
z_test2 = xtest[0:,0]
z_train1= X[0:,3]
z_train2= X[0:,0]
print (set(z_train2),set(z_train1),X.shape,xtest.shape)
# Visualize the data:

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[0] # training set size

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
    
    n_x = 39# size of input layer
    n_h = 39
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
    
    np.random.seed(9) 
    
    
    W1 = np.random.randn(17,17)
    W2 = np.random.randn(17,17)
    W3 = np.random.randn(17,1)
    
    
  #  assert (W1.shape == (n_h, n_x))
  #  assert (b1.shape == (n_h, 1))
  #  assert (W2.shape == (n_y, n_h))
  #  assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                #  "b1": b1,
                  "W2": W2,
                  "W3": W3
                  
                  }
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

def compute_cost(A2, Y, parameters,z1,z2,cache,X):
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
    X1 =[]
    X2 =[]
    for i in range(m):
            if (X[i][0]==0.4929843922796267):
                X1.append(X[i])
            else:
                X2.append(X[i])
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    #print (X1.shape)
    # Compute the cross-entropy cost
    z1=z1[:,np.newaxis]
    #print (z1.size)
    z2 =z2[:,np.newaxis] 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    ans1 = (sigmoid(np.dot(sigmoid(np.dot(sigmoid(np.dot(X1,W1)),W2)),W3)))
    count =0
    #print (ans1[789][0])
    """
    for i,j in zip(A):
            if (ans1[i][0]<0.5):
                count+=1
    print(count)
    """
    S1 = []
    S2 = []
    for i in range(m):
            if (X[i][3]==-0.8252521923879881):
                S1.append(X[i])
            else:
                S2.append(X[i])
    S1 = np.array(S1)
    S2 = np.array(S2)
    ans1=ans1[:,np.newaxis]
    ans2 = (sigmoid(np.dot(sigmoid(np.dot(sigmoid(np.dot(X2,W1)),W2)),W3)))
    count2 =0
    ansz1 = (sigmoid(np.dot(sigmoid(np.dot(sigmoid(np.dot(S1,W1)),W2)),W3)))
    ansz2 = (sigmoid(np.dot(sigmoid(np.dot(sigmoid(np.dot(S2,W1)),W2)),W3)))
    ans2 = np.array(ans2)

    for i in range(len(ansz1)):
        #print (ansz1[i])
        if ansz1[i]>0.5:
            count+=1
    for j in range(len(ansz2)):
        if ansz2[j]>0.5:
            count2+=1
    count3,count4 = 0,0
    for i in range(len(ans1)):
        if ans1[i]>0.5:
            count3+=1
    for i in range(len(ans2)):
        if ans2[i]>0.5:
            count4+=1

 #   ansz2=ansz2[:,np.newaxis]
#    ansz1=ansz1[:,np.newaxis]
    ans1 = count3/len(X1)
    ans2 = count4/len(X2)
    ansz1=count/len(S1)
    ansz2=count2/len(S2)
    #print (ans1,ans2,ansz1,ansz2)
    """
    if (ans1 ==0):
        ans1=1
    if( ans2 ==0):
        ans2=1
    if (ansz1 ==0):
        ansz1=1
    if (ansz2 ==0):
        ansz2=1
    """
    logprobs = 1*(np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y))+1*abs(ans2-ans1)**2#+abs(ansz2/ansz1-ansz1/ansz2)**2#0.2*abs(count/len(X1) - count2/len(X2))#,ans2[0]/ans1[0])
    cost = 1./m * np.nansum(logprobs)
   
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
 #   assert(isinstance(cost, float))
    
    return cost
  
import math
def backward_propagation(parameters, cache, X, Y,z1,z2):
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
    #print (W1.shape)
    # Retrieve also A1 and A2 from dictionary "cache".
    X1 =[]
    X2 =[]
    for i in range(m):
        if (X[i][0]==0.4929843922796267):
            X1.append(X[i])
        else:
            X2.append(X[i])
     
    X1 = np.array(X1)
    X2 = np.array(X2) 
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    #print (X1)
    X11 =np.dot(X1,W1)
    X12 = sigmoid(X11)
    X21 = np.dot(X12,W2)
    X22 = sigmoid (X21)
    X31 = np.dot(X22,W3)
    X32 = sigmoid(X31)
    
    #print (X32.shape)
    B11 = np.dot(X2,W1)
    B12 = sigmoid(B11)
    B21 = np.dot(B12,W2)
    B22 = sigmoid(B21)
    B31 = np.dot(B22,W3)
    B32 = sigmoid(B31)
    
    #print (B32.shape)
    z1=z1[:,np.newaxis]
    z2=z2[:,np.newaxis]
    cat = np.dot(np.transpose(X22),sigmoidPrime(X31))
    zat = np.dot(np.transpose(B22),sigmoidPrime(B31))
        
    dC3 =1/m*(cat -  zat)#+mat-sat)
    cat2 = np.dot(np.transpose(X12),sigmoidPrime(X21))
    zat2 = np.dot(np.transpose(B12),sigmoidPrime(B21))
    dC2 = 1/m*(cat2-zat2)#+mat2-sat2)

    cat3 =  np.dot(np.transpose(X1),sigmoidPrime(X11))
    zat3 =  np.dot(np.transpose(X2),sigmoidPrime(B11))
    #mat3= np.dot(np.transpose(S1),sigmoidPrime(C11))
    #sat3= np.dot(np.transpose(S2),sigmoidPrime(D11))
    dC1 =1/m*(cat3-zat3)#+mat3-sat3)
    #print (cat3.shape,zat3.shape)
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    covariance = cov(z1,A2)
   # a=0.6
    dZ3 = (A3-Y) #+ a*covariance**2
    dW3 =1/m* np.dot(np.transpose(A2),dZ3)
    #print (dC3.shape)
    dW3 = (1)*dW3+1.5*dC3
    dZ2 = np.dot(dZ3,np.transpose(W3))*sigmoidPrime(A3)
    dW2 =1/m* np.dot( np.transpose(A1),dZ2)
    dW2 = (1)*dW2+1.5*dC2
    dZ1 = np.dot(dZ2,np.transpose(W2))*sigmoidPrime(A2)#+(1 - np.power(A1,2))
    dW1 =1/m* np.dot( np.transpose(X),dZ1)
    dW1 = (1)*dW1+1.5*dC1
    grads = {"dW1": dW1,
             "dC2": dC2,
             "dW2": dW2,
             "dC1": dC1,
             "dW3": dW3,
             "dC3": dC3}
    
    return grads,covariance
  
  
 

def update_parameters(parameters, grads, covariance,learning_rate = 0.001):
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
    
    
    # Retrieve each gradient from the dictionary "grads"
   
    dW1 = grads["dW1"]
    dC1 = grads["dC1"]
    dW2 = grads["dW2"]
    dC2 = grads["dC2"]
    dW3 = grads["dW3"]
    dC3 = grads["dC3"]
    
    # Update rule for each parameter

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
  
  
def nn_model(X, Y,z1,z2, n_h, num_iterations = 100, print_cost=False):
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
    
    np.random.seed(99)
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
        cost = compute_cost(A2, Y, parameters,z1,z2,cache,X)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads,covariance = backward_propagation(parameters, cache, X, Y,z1,z2)
 
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
parameters = nn_model(X, Y,z_train1,z_train2, n_h = 16, num_iterations = 4000, print_cost=True)

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
#print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

#print (test_output.size)
#checking accuracy for differentnumber of hidden units
out = predict(parameters, xtest)
count_m=0
count_f=0
prob_f=0
prob_m=0
prob_fpr_m=0
prob_fpr_f=0

for i,j in zip(z_test1,Y):
    #print (z_test[i])
    if i==-0.8252521923879881 and j==1 :
        count_f+=1
    elif i ==1.2117507947556654 and j ==1:
        count_m+=1
 
count =0
for i in range(len(out)):
      if(out[i] > 0.50):
        out[i] =1
       # count+=1
      else:
        out[i] =0

      
for i,j,k in zip(out,z_test1,Y):
#    print (i,j)
    if j==-0.8252521923879881  and i ==1 and k==1:
        prob_f+=1
      #  print(j,i)-4.66909330348592
    elif j==1.2117507947556654 and i==1 and k==1:
        prob_m+=1

prob_f1,prob_m1,prob_f_dm,prob_m_dm,count_f2,count_m2=0,0,0,0,0,0
for i,j in zip(z_test2,Y):
    if i==0.4929843922796267 and j==1:
        count_f2+=1
      #  print(j,i)
    elif i==-2.0284617843089605 and j==1:
        count_m2+=1

for i,j,k in zip(out,z_test2,Y):
    if j==0.4929843922796267 and i==1 and k==1:
        prob_f_dm+=1
      #  print(j,i)
    elif i==-2.0284617843089605 and j==1:
        prob_m_dm+=1

for i,j in zip(out,z_test1):
#    print (i,j)
    if j==-0.8252521923879881  and i ==1:
        prob_f1+=1
      #  print(j,i)-4.66909330348592
    elif j==1.2117507947556654 and i==1:
        prob_m1+=1

prob_f2,prob_m2=0,0
for i,j in zip(out,z_test2):
    if j==0.4929843922796267  and i ==1:
        prob_f2+=1
      #  print(j,i)-4.66909330348592
    elif j==-2.0284617843089605 and i==1:
        prob_m2+=1

DI_n,DI_d= (prob_f1/count_f),(prob_m1/count_m)
#MT_n,MI_d = (prob_f),(prob_m)
print (((DI_n,DI_d)))
#print ("Disparate Imapct1",abs(DI_n-DI_d))
#print ("p% rule1",min(DI_n/DI_d,DI_d/DI_n))
DI_n2,DI_d2=(prob_f2/count_f2),(prob_m2/count_m2)
print (DI_n2,DI_d2)
print ("Disparate Imapct2",abs(DI_n2-DI_d2))
print ("p% rule2",min(DI_n2/DI_d2,DI_d2/DI_n2))
EO_n,EO_d = (prob_f/count_f),(prob_m/count_m)
print ("EO 1",abs(EO_n-EO_d))
EO_21,EO_22 = (prob_f_dm/count_f2),(prob_m_dm/count_m2)
print ("EO 2",abs(EO_21-EO_22))
accu = accuracy_score(ytest, out)*100
print('Test Acc: {:.4f}'.format(accu))
#cova = cov(z_test,out)
#print ("covarianve",cova)


# This may take about 2 minutes to run
print("--- %s seconds ---" % (time.time() - start_time))

