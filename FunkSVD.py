#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from math import sqrt

def FunkSVD(user_art_mat, latent_features=12, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    
    INPUT:
    user_art_mat - (numpy array) a matrix with users as rows, articles as columns, 
                   and whether or not user interacted with the article as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate 
    iters - (int) the number of iterations
    
    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    article_mat - (numpy array) a latent feature by movie matrix
    '''
    # Set up useful values to be used through the rest of the function
    n_users = user_art_mat.shape[0]
    n_articles = user_art_mat.shape[1]
    num_interact = np.count_nonzero(user_art_mat)
    
    # initialize the user and movie matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    article_mat = np.random.rand(latent_features, n_articles)  
    
    # initialize sse at 0 for first iteration
    sse_accum = 0
    
    # keep track of iteration and MSE
    #print("Optimizaiton Statistics")
    #print("Iterations | Mean Squared Error ")
    
    for iter in range(iters):
        # update our sse
        old_sse = sse_accum
        sse_accum = 0
        
        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_articles):
                # if the rating exists
                if user_art_mat[i, j] > 0:
                    
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = user_art_mat[i, j] - np.dot(user_mat[i, :], article_mat[:, j])
                    
                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2
                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*article_mat[k, j])
                        article_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

        #print("%d \t\t %f" % (iter+1, sse_accum / num_interact))   
    return user_mat, article_mat

def user_item_train_test_split (df_joint, test_size=0.2, random_seed = 22):
    """Function that prepares train and test user_item_matrices out of df_joint.
    
    INPUT: 
    1.  dataframe with interact, should including following columns: user_id, article_id, interact 
    2.  test_size, default = 0.2
    3.  random_seed, default = 22
    OUTPUT:
    1.  train user_item_matrix
    2.  test user_item_matrix
        
    """
    
    #saving a df to work with
    copy_df = df_joint.copy ()
    
    data_matrix = copy_df.groupby(['user_id', 'article_id'])['interact'].sum ().unstack()
    
    #taking part of ratings 
    test_matrix = data_matrix.sample(frac = test_size, random_state = random_seed)
    
    #eliminating taken data from rate_df so to form train
    train_matrix = data_matrix.drop (index = test_matrix.index)
    
    return train_matrix, test_matrix 


def validation_func (test_matrix, predict_train_matrix, is_train =False, verbose=True):
    """Function that runs validation of predicted user item matrix (predict_train_matrix) on the given user item matrix 
    (test or train matrix). 
    
    INPUT: 
    Test user item matrix, predicted user item matrix, is_train sign (if validation done on train), verbose sign (whether to         print results)
    OUTPUT: 
    MSE error value, Accuracy value. If verbose equals True then results just printed. Nothing returned.
       
    """
        
    #will be looping over common customers only
    test_idx = test_matrix.index.intersection (predict_train_matrix.index)
    test_cols = test_matrix.columns

    sse_accum = 0
    diff = 0
    acc = []

    for i in test_idx: 
        for j in test_cols: 
            if test_matrix.at [i,j]>=0:
                # compute the error as the actual minus the dot product of the user and offer latent features
                diff = test_matrix.at [i,j] - predict_train_matrix.at [i,j]
                        
                # Keep track of the sum of squared errors for the matrix
                sse_accum += diff**2
                acc.append (abs (diff))

    #deviding on the overall not null values in test_matrix
    mse_error = sse_accum/(test_matrix.notnull().sum ().sum())
    acc_test = 1 - sum (acc)/(test_matrix.notnull().sum ().sum())
    
    if (is_train == False)&(verbose): 
        print ('\n')
        print ('MSE on the test set is: ', mse_error)
        print ('RMSE on the test set is: ', sqrt(mse_error))
        print ('Accuracy on the test set is: ', acc_test)
        print ('\n')
    if (is_train == True)&(verbose): 
        print ('\n')
        print ('MSE on the train set is: ', mse_error)
        print ('RMSE on the train set is: ', sqrt(mse_error))
        print ('Accuracy on the train set is: ', acc_test)
        print ('\n')
    #if no need for printing just return values of MSE and Accuracy
    if verbose == False: 
        return mse_error, acc_test
