import numpy as np

class DataModel(object):
    '''
    Base data model.
    '''
    def __init__(self):
        self.p, self.d = self.Phi.shape

        self._diagonalise()
        self._commute()

    def get_info(self):
        info = {
            'data_model': 'base',
            'latent_dimension': self.d,
            'input_dimension': self.p
        }
        return info

    def _check_commute(self):
        if np.linalg.norm(self.Omega @ self.PhiPhiT - self.PhiPhiT @ self.Omega) < 1e-10:
            self.commute = True
        else:
            self.commute = False
            self._UTPhiPhiTU = np.diagonal(self.eigv_Omega.T @ self.PhiPhiT @ self.eigv_Omega)

    def _diagonalise(self):
        '''
        Diagonalise covariance matrices.
        '''
        self.spec_Omega, self.eigv_Omega = np.linalg.eigh(self.Omega)
        self.spec_Omega = np.real(self.spec_Omega)

        self.spec_PhiPhit = np.real(np.linalg.eigvalsh(self.PhiPhiT))
