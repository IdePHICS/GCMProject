import numpy as np
from .base_data_model import DataModel

class Custom(DataModel):
    '''
    Custom allows for user to pass his/her own covariance matrices.
    -- args --
    teacher_teacher_cov: teacher-teacher covariance matrix (Psi)
    student_student_cov: student-student covariance matrix (Omega)
    teacher_student_cov: teacher-student covariance matrix (Phi)
    '''
    def __init__(self, *, teacher_teacher_cov, student_student_cov, teacher_student_cov):
        self.Psi = teacher_teacher_cov
        self.Omega = student_student_cov
        self.Phi = teacher_student_cov

        self.p, self.k = self.Phi.shape
        self.PhiPhiT = self.Phi @ self.Phi.T
        self.rho = np.trace(self.Psi) / self.k

        self._check_sym()
        self._diagonalise() # see base_data_model
        self._check_commute()

    def get_info(self):
        info = {
            'data_model': 'custom',
            'teacher_dimension': self.k,
            'student_dimension': self.p
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
