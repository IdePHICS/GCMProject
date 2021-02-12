import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
def get_samples(*, samples, generator, teacher_map, student_map, 
                teacher_weights, mean_v, mean_u, task):
    
    latent_dim = generator.input_dim
    with torch.no_grad():
        Z = torch.randn(samples, latent_dim, 1, 1).to(device)
        
        img = generator(Z)
        img = img.view(samples, -1)

    V = student_map(img).reshape(samples, -1)
    U = teacher_map(img).reshape(samples, -1)
    
    p = U.shape[1]
    d = V.shape[1]

    V -= mean_v
    U -= mean_u
    
    # Compute labels
    y = U @ teacher_weights / np.sqrt(p)
    if task == 'classification':
        y = np.sign(y)

    return V / np.sqrt(d), y

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
        return ridge_estimator(X, y,lamb=lamb)
    
    elif loss == 'logistic':
        return LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, 
                                  C = lamb**(-1), max_iter=1e4, tol=1e-7, verbose=0).fit(X, y).coef_[0]
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
    
def get_errors(seeds=10, loss='l2', task='regression', *, 
               samples, lamb, generator, teacher_map, student_map, 
               teacher_weights, mean_v, mean_u):
    '''
    Get averaged training and test error over a number of seeds for a fixed 
    number of samples.
    '''
    
    eg, et = [], []
    p = len(mean_v)
    for i in range(seeds):
        print('Seed: {}'.format(i))

        # Randomly subsample a set of indices (without replacement)
        X_train, y_train = get_samples(samples = samples, 
                                       generator = generator,
                                       teacher_map = teacher_map, 
                                       student_map = student_map, 
                                       teacher_weights = teacher_weights, 
                                       mean_v = mean_v, 
                                       mean_u = mean_u, 
                                       task = task)

        X_test, y_test = get_samples(samples = samples, 
                                     generator = generator,
                                     teacher_map = teacher_map, 
                                     student_map = student_map, 
                                     teacher_weights = teacher_weights, 
                                     mean_v = mean_v, 
                                     mean_u = mean_u, 
                                     task = task)
        
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
             seeds = 10, *, loss, task, teacher_map, student_map, 
             teacher_weights, mean_v, mean_u, generator):

    data = {'test_error': [], 'train_loss': [], 'test_error_std': [], 
            'train_loss_std': [], 'lambda': [], 
            'sample_complexity': [], 'samples': [], 'task': [], 'loss': []}

    
    p = len(mean_v)
    k = len(teacher_weights)
    gamma = k/p
    
    for alpha in sc_range:
        print('Simulating sample complexity: {}'.format(alpha))
        samples = int(alpha * p)
        et, et_std, eg, eg_std = get_errors(samples = samples,
                                            lamb = lamb, 
                                            seeds = seeds, 
                                            task = task, 
                                            loss = loss,
                                            generator = generator,
                                            teacher_map = teacher_map, 
                                            student_map = student_map, 
                                            teacher_weights = teacher_weights, 
                                            mean_v = mean_v, 
                                            mean_u = mean_u)
        
        data['sample_complexity'].append(alpha)
        data['samples'].append(samples)

        data['lambda'].append(lamb)
        data['loss'].append(loss)
        data['task'].append(task)

        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_loss'].append(et)
        data['train_loss_std'].append(et_std)

    return pd.DataFrame.from_dict(data)