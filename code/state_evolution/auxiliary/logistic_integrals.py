'''
Auxiliary integrals needed for computing the likelihood update functions
for logistic regression.
'''

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar
import mpmath

def xstar(x, y,omega,V):
    return (x-omega)**2/(2*V) + np.log(1+np.exp(-x*y))

def f_mhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: xstar(x, 1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_plus - ω)

def f_mhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: xstar(x, -1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_minus - ω)

def integrate_for_mhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_mhat_plus(ξ, M, Q, V, Vstar) * np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    I2 = quad(lambda ξ: f_mhat_minus(ξ, M, Q, V, Vstar) * np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    return (I1 - I2)*(1/np.sqrt(2*np.pi*Vstar))

# Vhat_x #
def f_Vhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: xstar(x, 1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2))) * (1 + erf(ωstar/np.sqrt(2*Vstar)))

def f_Vhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: xstar(x, -1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2))) * (1 - erf(ωstar/np.sqrt(2*Vstar)))
def integrate_for_Vhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_Vhat_plus(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    I2 = quad(lambda ξ: f_Vhat_minus(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    return (1/2) * (I1 + I2)

# Qhat_x#
def f_qhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: xstar(x, 1, ω, V))['x']
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * (λstar_plus - ω)**2

def f_qhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: xstar(x, -1, ω, V))['x']
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * (λstar_minus - ω)**2

def integrate_for_Qhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_qhat_plus(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    I2 = quad(lambda ξ: f_qhat_minus(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    return (1/2) * (I1 + I2)

def Integrand_training_error_plus_logistic(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = np.float(mpmath.findroot(lambda λstar_plus: λstar_plus - ω - V/(1 + np.exp(np.float(λstar_plus))), 10e-10))
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * np.log(1 + np.exp(-λstar_plus))

def Integrand_training_error_minus_logistic(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = np.float(mpmath.findroot(lambda λstar_minus: λstar_minus - ω + V/(1 + np.exp(-np.float(λstar_minus))), 10e-10))
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * np.log(1 + np.exp(λstar_minus))

def traning_error_logistic(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_plus_logistic(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    I2 = quad(lambda ξ: Integrand_training_error_minus_logistic(ξ, M, Q, V, Vstar)* np.exp(-ξ**2/2) * (1/np.sqrt(2*np.pi)), -10, 10)[0]
    return (1/2)*(I1 + I2)
