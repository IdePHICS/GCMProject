import numpy as np
from .base_model import Model


class L2Classification(Model):
    '''
    Implements updates for a classification with square error.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation

        self.data_model = data_model

    def get_info(self):
        info = {
            'model': 'l2_classification',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                        mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                        (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit /
                                                    (self.lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU *
                                   self.data_model.spec_Omega /
                                   (self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/
                                                (self.lamb + Vhat * self.data_model.spec_Omega))


        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        Vhat = self.alpha * 1/(1+V)
        qhat = self.alpha * (1 + q - 2*m*np.sqrt(2/(np.pi*self.data_model.rho))) / (1+V)**2
        mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(1+V) * np.sqrt(2/(np.pi*self.data_model.rho))

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)


    def get_test_error(self, q, m):
        return  np.arccos(m/np.sqrt(self.data_model.rho * q)) / np.pi

    def get_train_loss(self, V, q, m):
        return .25*(1 + q - 2*m*np.sqrt(2/(np.pi*self.data_model.rho))) / (1+V)**2
