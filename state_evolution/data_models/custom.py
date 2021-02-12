import numpy as np
from .base_data_model import DataModel

class Custom(DataModel):
    '''
    Custom allows for user to pass his/her own covariance matrices.
    -- args --
    teacher_teacher_cov: teacher-teacher covariance matrix (Psi)
    student_student_cov: student-student covariance matrix (Omega)
    teacher_student_cov: teacher-student covariance matrix (Phi)
    teacher_weights: teacher weight vector (theta0)
    '''
    def __init__(self, *, teacher_teacher_cov, student_student_cov, 
                 teacher_student_cov, teacher_weights):
        
        self.Psi = teacher_teacher_cov
        self.Omega = student_student_cov
        self.Phi = teacher_student_cov.T
        self.theta = teacher_weights
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        
        self.PhiPhiT = (self.Phi @ self.theta.reshape(self.k,1) @ 
                        self.theta.reshape(1,self.k) @ self.Phi.T)
        
        self.rho = self.theta.dot(self.Psi @ self.theta) / self.k

        self._check_sym()
        self._diagonalise() # see base_data_model
        self._check_commute()

    def get_info(self):
        info = {
            'data_model': 'custom',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info

    def _check_sym(self):
        '''
        Check if input-input covariance is a symmetric matrix.
        '''
        if (np.linalg.norm(self.Omega - self.Omega.T) > 1e-5):
            print('Student-Student covariance is not a symmetric matrix. Symmetrizing!')
            self.Omega = .5 * (self.Omega+self.Omega.T)

        if (np.linalg.norm(self.Psi - self.Psi.T) > 1e-5):
            print('Teacher-teaccher covariance is not a symmetric matrix. Symmetrizing!')
            self.Psi = .5 * (self.Psi+self.Psi.T)


class CustomSpectra(DataModel):
    '''
    Custom allows for user to pass directly the spectra of the covarinces.
    -- args --
    spec_Psi: teacher-teacher covariance matrix (Psi)
    spec_Omega: student-student covariance matrix (Omega)
    diagonal_term: projection of student-teacher covariance into basis of Omega
    '''
    def __init__(self, *, rho, spec_Omega, diagonal_term, gamma):
        self.rho = rho
        self.spec_Omega = spec_Omega
        self._UTPhiPhiTU = diagonal_term

        self.p = len(self.spec_Omega)
        self.gamma = gamma
        self.k = int(self.gamma * self.p)

        self.commute = False

    def get_info(self):
        info = {
            'data_model': 'custom_spectra',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info
