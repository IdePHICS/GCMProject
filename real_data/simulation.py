import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def ridge_estimator(X, y, lamb=0.1):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y

def get_estimator(X,y, lamb=0.1, loss='l2'):
    _, p = X.shape
    if loss == 'l2':
        return ridge_estimator(X/np.sqrt(p), y,lamb=lamb)
    
    elif loss == 'logistic':
        return LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, 
                                  C = lamb**(-1), max_iter=1e4, tol=1e-7, verbose=0).fit(X/np.sqrt(p),y).coef_[0]
    
def mse(y,yhat):        
    return np.mean((y-yhat)**2)

def logistic_loss(z):
    return np.log(1+np.exp(-z))

def predict_label(X, estimator, task='regression'):
    _, p = X.shape
    
    if task == 'regression':
        return X @ estimator / np.sqrt(p)
    elif task == 'classification':
        return np.sign(X @ estimator / np.sqrt(p))
    
def get_errors(sample_complexity, X, y, lamb, seeds=10, 
               loss='l2', task='regression'):
    '''
    Get averaged training and test error over a number of seeds for a fixed 
    number of samples.

    Input:
        - sample_complexity: sample complexity n/p
        - X: full data set (train+test)
        - y: labels of the total data set
        - seeds: number of seeds
        - task: regression or classification
        - loss: loss function
    Return:
        - (avg train error, std train error, avg test error, std test error)
    '''
    
    eg, et = [], []
    for i in range(seeds):
        print('Seed: {}'.format(i))

        _, p = X.shape

        # Randomly subsample a set of indices (without replacement)
        samples = int(sample_complexity * p) # number of samples
        inds = np.random.choice(range(X.shape[0]), size=samples, replace=False)

        # Subsample from whole universe
        X_train = X[inds, :] # training data
        y_train = y[inds] # training labels

        # Compute test error on the whole universe
        X_test = X # test data
        y_test = y # test labels
        
        # Compute estimator
        w = get_estimator(X_train, y_train, lamb=lamb, loss = loss)

        # Label estimate
        yhat_train = predict_label(X_train, w, task=task)
        yhat_test = predict_label(X_test, w, task=task)

        # Errors
        test_error = mse(y_test, yhat_test)
        train_error = mse(y_train, yhat_train)

        if task == 'classification':
            test_error *= 0.25
            train_error *= 0.25
        
        if loss == 'logistic':
            train_error = np.mean(logistic_loss((X_train @ w)/np.sqrt(p) * y_train))
            
        eg.append(test_error)
        et.append(train_error)

    return (np.mean(et), np.std(et), np.mean(eg), np.std(eg))

def simulate(sc_range = np.linspace(0.1, 2.5, 10), lamb = 0.01, 
             seeds = 10, *, X, y, loss, task):
    '''
    - Inputs:
    sc_range: range of sample complexity
    X: Full data set
    y: Full labels
    loss: logistic or l2
    task: regression or classification

    - Return:
    data: pandas datafram with learning curves
    '''
    data = {'test_error': [], 'train_loss': [], 'test_loss_std': [], 
            'train_error_std': [], 'lambda': [], 'sample_complexity': [], 
            'task': [], 'loss': []}
    
    for alpha in sc_range:
        print('Simulating sample complexity: {}'.format(alpha))
        et, et_std, eg, eg_std = get_errors(alpha, X, y, lamb, seeds=seeds, 
                                            task=task, loss=loss)

        data['sample_complexity'].append(alpha)
        data['lambda'].append(lamb)
        data['loss'].append(loss)
        data['task'].append(task)

        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_loss'].append(et)
        data['train_loss_std'].append(et_std)

    return pd.DataFrame.from_dict(data)