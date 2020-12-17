class Model(object):
    '''
    Base class for a model.
    -- args --
    sample_complexity: sample complexity
    regularisation: ridge penalty coefficient
    data_model: data_model instance. See /data_model/
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation
        self.data = data_model

        self.parameters, self.dimension = self.data_model.Phi.shape
        self.gamma = self.dimension / self.parameters

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'model': 'generic',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def update_se(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError

    def get_test_error(self, q, m):
        '''
        Method for computing the test error from overlaps.
        '''
        raise NotImplementedError

    def get_train_loss(self, V, q, m):
        '''
        Method for computing the training loss from overlaps.
        '''
        raise NotImplementedError
