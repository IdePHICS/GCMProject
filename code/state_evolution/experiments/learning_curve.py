from ..data_models.custom import Custom
from ..models.ridge_regression import RidgeRegression
from ..models.logistic_regression import LogisticRegression
from ..algorithms.state_evolution import StateEvolution

class CustomExperiment(object):
    '''
    Implements experiment for generic task and custom, for fixed
    regularisation.

    Note sample complexity is passed to run_experiment as an argument
    allowing for running several sample complexities for the same pre-diagonalised
    data model.
    '''
    def __init__(self,initialisation='uninformed', tolerance=1e-10, damping=0,
                       verbose=False, max_steps=1000, *,
                       task, regularisation, data_model):

        self.task = task
        self.lamb = regularisation
        self.data_model = data_model
        # Hyperparameters
        self.initialisation=initialisation,
        self.tolerance=tolerance
        self.damping=damping,
        self.verbose=verbose,
        self.max_steps=max_steps


    def learning_curve(self, *, alphas):
        curve = {
            'task': [],
            'gamma': [],
            'rho': [],
            'sample_complexity': [],
            'V': [],
            'm': [],
            'q': [],
            'test_error': [],
            'train_loss': [],
        }
        for alpha in alphas:
            self._run(sample_complexity=alpha)
            info_sp = self.se.get_info()
            info_data = self.data_model.get_info()

            curve['task'].append(self.task)
            curve['lambda'].append(self.lamb)
            curve['gamma'].append(info_data['teacher_dimension'] / info_data['student_dimension'])
            curve['rho'].append(self.data_model.rho)

            curve['test_error'].append(info_sp['test_error'])
            curve['train_loss'].append(info_sp['train_loss'])
            curve['V'].append(info_sp['overlaps']['variance'])
            curve['q'].append(info_sp['overlaps']['self_overlap'])
            curve['m'].append(info_sp['overlaps']['self_overlap'])

        self.learning_curve = pd.DataFrame.from_dict(curve)

    def _run(self, *, sample_complexity):
        '''
        Runs saddle-point equations.
        '''
        self._initialise_model(sample_complexity)

        self.se = StateEvolution(model=self.model,
                       initialisation=self.initialisation,
                       tolerance=self.tolerance,
                       damping=self.damping,
                       verbose=self.verbose,
                       max_steps=self.max_steps)

        self.se.iterate()

    def _initialise_model(self, sample_complexity):
        if self.task == 'ridge_regression':
            self.model = RidgeRegression(sample_complexity = sample_complexity,
                                         regularisation=self.lamb,
                                         data_model = self.data_model)


        elif self.task == 'logistic_regression':
            self.model = LogisticRegression(sample_complexity = sample_complexity,
                                            regularisation=self.lamb,
                                            data_model = self.data_model)
        else:
            print('{} not implemented.'.format(self.task))

    def save_experiment(self, data_dir):
        '''
        Saves result of experiment in .json file with info for reproductibility.
        '''
        from datetime import datetime
        import os
        import json
        import uuid

        unique_id = uuid.uuid4().hex
        day, time = datetime.now().strftime("%d_%m_%Y"), datetime.now().strftime("%H:%M:%S")

        sub_dir = '{}/{}'.format(data_dir, day)
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)

        name = '{}/{}_{}.json'.format(sub_dir, self.task, unique_id)
        print('Saving experiment at {}'.format(name))
        self.learning_curve.to_csv(outfile, index=False)
