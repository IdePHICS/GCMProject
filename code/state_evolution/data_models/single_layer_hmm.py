import numpy as np
from .base_data_model import DataModel

NL = {'relu', 'sign', 'tanh', 'erf'}
COEFICIENTS = {'relu': (1/np.sqrt(2*np.pi), 0.5, np.sqrt((np.pi-2)/(4*np.pi))),
               'erf': (0, 2/np.sqrt(3*np.pi), 0.200364), 'tanh': (0, 0.605706, 0.165576),
               'sign': (0, np.sqrt(2/np.pi), np.sqrt(1-2/np.pi))}

class SingleLayerHMM(DataModel):
    '''
    Load data from a single-layer hidden-manifold model.
    -- args --
    destribution: distribution of first layer fixed weights (a.k.a. projections).
    gaussian: i.i.d. Gaussian with zero-mean and variance 1.
    latent_dimension: dimension of latent manifold.
    teacher_dimension: dimension of the teacher.
    student_dimension: dimension of the student.
    activation: activation function / non-linearity. Available: relu, erf, sign.
    '''
    def __init__(self, distribution='gaussian', *, teacher_dimension, student_dimension,
                 latent_dimension, teacher_activation, student_activation):

        self.d = latent_dimension
        self.k = teacher_dimension
        self.p = student_dimension
        self.distribution = distribution
        self.teacher_activation = teacher_activation
        self.student_activation = student_activation

        self._generate_matrices()

        self._diagonalise() # see base_data_model
        self.commute = False

    def get_info(self):
        info = {
            'data_model': 'single_layer_hmm',
            'teacher_activation': self.teacher_activation,
            'student_activation': self.student_activation,
            'projection': self.distribution,
            'teacher_dimension': self.teacher_dimension,
            'student_dimension': self.student_dimension,
            'latent_dimension': self.latent_dimension,
        }
        return info

    def _sample_F(self):
        '''
        Sample the fixed weight matrix.
        '''
        if self.distribution == 'gaussian':
            self.projection = {
            'teacher': np.random.normal(0,1,(self.d, self.k)) / np.sqrt(self.d),
            'student': np.random.normal(0,1,(self.d, self.p)) / np.sqrt(self.d)
            }
        else:
            print('{} matrices not implemented.'.format(self.distribution))

    def _get_coefficients(self):
        '''
        Get coefficients for first layer activations.
        '''
        if (self.teacher_activation in NL) and (self.student_activation in NL):
            coefficients = {
                'teacher': COEFICIENTS[self.teacher_activation],
                'student': COEFICIENTS[self.student_activation],
            }
            return coefficients
        else:
            print('{} or {}activation not implemented.'.format(self.teacher_activation, self.student_activation))

    def _generate_matrices(self):
        '''
        Generate input-input and latent-input covariances from weights and
        coefficients.
        '''
        self._sample_F()
        coefficients = self._get_coefficients()

        self.Omega = (coefficients['student'][1]**2 * self.projection['student'].T @ self.self.projection['student'] +
                      coefficients['student'][2]**2 * np.identity(self.p))
        self.Psi = (coefficients['teacher'][1]**2 * self.projection['teacher'].T @ self.self.projection['teacher'] +
                      coefficients['teacher'][2]**2 * np.identity(self.p))

        self.Phi = (coefficients['teacher'][1]*coefficients['student'][1] *
                    self.projection['student'].T @ self.projection['teacher'])

        self.PhiPhiT = self.Phi @ self.Phi.T
        self.rho = np.trace(self.Psi) / self.k
