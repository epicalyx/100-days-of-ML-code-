#!/usr/bin/env python
# coding: utf-8

# # Polynomial Ridge Regression 

# In[25]:


'''  Helpful resources: Coursera ML course ,
http://kldavenport.com/regularized-logistic-regression-intuition/,
https://medium.com/@k_dasaprakash/logistic-regression-5b371cc0824f,
http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html'''


# In[20]:


#importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures


# In[21]:


#logistic(sigmoid) function 
def sigmoid_function(z):
    return 1/(1+np.exp(-z))

#feature mapping for polynomial regression
def mapfeature(X,order):
    poly=PolynomialFeatures(order)
    return poly.fit_transform(X)


# In[22]:


# cost function J of theta
def cost_function(theta,h,y,lamda):
    regularization_term=(float(lamda)/2)*(theta**2)
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean() +sum(regularization_term[1:])/y.size        # lamda= regularization factor


#gradient descent algorithm for logistic regression
def logistic_reg(alpha,X,y,lamda,max_iterations=1500):
    converged=False
    iterations=0
    theta=np.zeros(X.shape[1])
    
    while not converged:
        z=np.dot(X,theta)
        h=sigmoid_function(z)
        gradient=np.dot(X.T,(h-y))/y.size + (float(lamda)*theta/y.size).T
        theta=theta-alpha*gradient
        
        z=np.dot(X,theta)
        h=sigmoid_function(z)
        e=cost_function(theta,h,y,lamda)
        print("Cost function J=",e)
        J=e
        
        iterations+=1
        
        if iterations==max_iterations:
            print("Maximum iterations exceeded!")
            print("Optimal value of cost function J=",J)
            converged = True
            
    return theta


# In[23]:


# function to plot data
def plotData(X,y):
    plt.scatter(X[:,0],X[:,1],c=y,s=118,alpha=0.5,cmap='coolwarm')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()
    
# function to plot decision boundary    
def plotDecisionBoundary(X,y,theta,order):
    plt.scatter(X[:,0],X[:,1],c=y,s=118,alpha=0.5,cmap='coolwarm')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    dim=np.linspace(-1.0,1.5,800)
    x,y=np.meshgrid(dim,dim)
    poly=mapfeature(np.column_stack((x.flatten(),y.flatten())),order)
    z=(np.dot(poly,theta)).reshape(800,800)
    plt.contour(x,y,z,levels=[0],colors=['r'])
    plt.show()


# In[24]:


if __name__=='__main__':
    df=pd.read_csv('microchip.txt',sep=',',header=None)
    df=df.sample(frac=1).reset_index(drop=True)
    
    X=df.iloc[:, :-1]
    y=df.iloc[:,-1]
    
    X=np.array(X)
    y=np.array(y)
    
    plotData(X,y) 
    
    X_poly=mapfeature(X,order=6)
    alpha=0.01
    lamda=0.04
    
    theta=logistic_reg(alpha,X_poly,y,lamda,max_iterations=1500)
    print(theta)
    
    plotDecisionBoundary(X,y,theta,order=6)

