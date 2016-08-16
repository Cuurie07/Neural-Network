import numpy as np
#np.set_#printoptions(threshold=np.inf)
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt




def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return np.reciprocal((1+ np.exp(-z)))
    

def preprocess():
    
    mat=loadmat('mnist_all.mat')
    traindata=mat.get('train0')
    traindata=traindata.astype('float')
    traindata=np.divide(traindata,255)
    tup=traindata.shape
    rows=tup[0]
    trainlabel=np.zeros((np.array((rows)),1))
    trainlabel.fill(0)
   


    #Pick a reasonable size for validation data

    for i in range(1,10):
        data=mat.get('train'+str(i))
        data=data.astype('float')
        data=np.divide(data,255)
        c=data.shape
        row=c[0]
        b=np.zeros((np.array((row)),1))
        b.fill(i)
        trainlabel=np.concatenate((trainlabel,b))
        traindata=np.concatenate((traindata,data))


    testdata=mat.get('test0')
    testdata=testdata.astype('float')
    testdata=np.divide(testdata,255)

    tu=testdata.shape
    ro=tu[0]
    test_label=np.zeros((np.array((ro)),1))
    test_label.fill(0)
   

    for i in range(1,10):
        tdata=mat.get('test'+str(i))
        tdata=tdata.astype('float')
        tdata=np.divide(tdata,255)
        d=tdata.shape
        r=d[0]
        t=np.zeros((np.array((r)),1))
        t.fill(i)
        test_label=np.concatenate((test_label,t))
        testdata=np.concatenate((testdata,tdata))

    fearray=np.all(traindata == traindata[0,:], axis = 0)
    for i in range (0,784):
        if (fearray[i]):
              traindat=traindata.compress(np.logical_not(fearray), axis=1)
              test_data=testdata.compress(np.logical_not(fearray), axis=1)

    a1 = range(traindata.shape[0])
    aperm = np.random.permutation(a1)
    train_data = traindat[aperm[0:50000],:]
    validation_data = traindat[aperm[50000:],:]


    train_label = trainlabel[aperm[0:50000],:]
    validation_label = trainlabel[aperm[50000:],:]
    train_label= np.ravel(train_label)
    test_label=np.ravel(test_label)
    validation_label=np.ravel(validation_label)

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    

    traininglabel=np.zeros((training_label.shape[0],10),dtype=float)
    
    
    for z in range (0,training_label.shape[0]):
        num=training_label[z]    
        traininglabel[z][num]=1    
         
    dplus1=n_input+1
    mplus1=n_hidden+1
    
    
    onevector=np.ones((training_data.shape[0],1))
    
    train=np.hstack((training_data,onevector))
    
    #transtrain=np.transpose(train)
    
    aj=np.dot(train,np.transpose(w1))
    
    zj=sigmoid(aj)
    
    onevec=np.ones((zj.shape[0],1))
    
    newzj=np.hstack((zj,onevec)) 
    
    bl=np.dot(newzj,np.transpose(w2))
    
    ol=sigmoid(bl)
    
    #end of feedforward pass
     
    yo=(traininglabel-ol)**2
    
    jp=yo*(0.5)
    
    finsum=np.sum(jp) 
    
    n=training_label.shape[0]
    
    jvalue=finsum/n
    
    #objval after error propagation
     
    sum1=0
    sum2=0
    obj_val = 0  
    lamby=lambdaval/(2*training_data.shape[0])
    
    # J with regularization
    
    #for j in range(n_hidden):
        #for i in range(dplus1):
    sum1=np.sum(w1**2)
    
            
    #for l in range(n_class):
       # for j in range(mplus1):
    sum2=np.sum(w2**2)
        
    
    finalsum=sum1+sum2
    finalmul=lamby*finalsum
    
    obj_val=jvalue+finalmul
    
    #till here we have calculated obj_value
    #out=open('/Users/anjali/Desktop/output.txt','w')
    #onev=np.ones((training_label.shape[0],n_class), dtype=float)
    
 
    #oneol=np.subtract(onev,ol)
    
    #oneolol=np.dot(oneol,np.transpose(ol))
 
    #yol=np.subtract(traininglabel,ol)

    touse=(traininglabel-ol)*(1-ol)*ol

    minuszj=newzj*(-1)

    gradjp2=np.dot(np.transpose(touse),minuszj)


    lamw2=w2*(lambdaval)
  
    #sumingradj2= np.add(gradjp2,lamw2)
    
    sumingradj2=gradjp2+lamw2
    gradj2= sumingradj2/n
    
    #out.close()
    
    #on=np.ones((training_data.shape[0],n_hidden), dtype=float)
   
    #onezj=np.subtract(on,zj)
 
    #minusonezj=np.multiply(onezj,-1)
    
    #colu=w2.shape[1]-1
    #neww2=np.delete(w2,colu,1)
    
    
    term1=(1-newzj)*newzj*(-1)
    
    term2=np.dot(touse,w2)*term1
    
    gradjp1=np.dot(np.transpose(term2),train)
    gradjp1=gradjp1[:-1,:]
   
    lamw1=w1*lambdaval
    
    gradsum=gradjp1+lamw1
  
    
    gradj1=gradsum/n
   
     
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #gradj1=gradj1+lambdaval*w1
    #gradj2=gradj2+lambdaval*w2
    obj_grad = np.concatenate((gradj1.flatten(), gradj2.flatten()),0)
    
    obj_grad.astype
     
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    
    labels = np.array([])
    #Your code here

    data_bias = np.ones((data.shape[0], 1))

    # Hidden Bias
    data = np.hstack((data, data_bias))
    net1 = np.dot(data, np.transpose(w1))

    # Hidden Layer output
    out1 = sigmoid(net1)

    out1_bias = np.ones((out1.shape[0], 1))
    out1 = np.hstack((out1, out1_bias))

    # Final output
    labels = sigmoid(np.dot(out1, np.transpose(w2)))

    # The prediction is the index of the output unit with the max o/p
    labels = np.argmax(labels, axis=1)

    return labels
   



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = 50;
n_hidden=20
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.6;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:'  + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
