from ..data_models.custom import Custom
from ..models.ridge_regression import RidgeRegression
from ..models.l2_classification import L2Classification
from ..models.logistic_regression import LogisticRegression
from ..algorithms.state_evolution import StateEvolution
import pandas as pd

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
        self.initialisation=initialisation
        self.tolerance = tolerance
        self.damping = damping
        self.verbose = verbose
        self.max_steps = max_steps


    def learning_curve(self, *, alphas):
        curve = {
            'task': [],
            'gamma': [],
            'lambda': [],
            'rho': [],
            'sample_complexity': [],
            'V': [],
            'm': [],
            'q': [],
            'test_error': [],
            'train_loss': [],
        }
        for alpha in alphas:
            if self.verbose:
                print('Runninig sample complexity: {}'.format(alpha))

            self._run(sample_complexity = alpha)
            info_sp = self.se.get_info()
            info_data = self.data_model.get_info()

            curve['task'].append(self.task)
            curve['gamma'].append(info_data['teacher_dimension'] / info_data['student_dimension'])
            curve['lambda'].append(self.lamb)
            curve['rho'].append(self.data_model.rho)
            curve['sample_complexity'].append(alpha)

            curve['test_error'].append(info_sp['test_error'])
            curve['train_loss'].append(info_sp['train_loss'])
            curve['V'].append(info_sp['overlaps']['variance'])
            curve['q'].append(info_sp['overlaps']['self_overlap'])
            curve['m'].append(info_sp['overlaps']['teacher_student'])

        self._learning_curve = pd.DataFrame.from_dict(curve)


    def get_curve(self):
        return self._learning_curve

    def _run(self, *, sample_complexity):
        '''
        Runs saddle-point equations.
        '''
        self._initialise_model(sample_complexity)

        self.se = StateEvolution(model=self.model,
                       initialisation=self.initialisation,
                       tolerance=self.tolerance,
                       damping=self.damping,
                       verbose=False,
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

        elif self.task == 'l2_classification':
            self.model = L2Classification(sample_complexity = sample_complexity,
                                            regularisation=self.lamb,
                                            data_model = self.data_model)

        else:
            print('{} not implemented.'.format(self.task))

    def save_experiment(self, date=False, unique_id=False, directory='.', *, name):
        '''
        Saves result of experiment in .json file with info for reproductibility.
        '''
        path = '{}/{}'.format(directory, name)

        if date:
            from datetime import datetime
            day, time = datetime.now().strftime("%d_%m_%Y"), datetime.now().strftime("%H:%M")
            path += '_{}_{}'.format(day, time)

        if unique_id:
            import uuid
            unique_id = uuid.uuid4().hex
            path += '_{}'.format(unique_id)

        path += '.csv'
        print('Saving experiment at {}'.format(path))
        self._learning_curve.to_csv(path, index=False)
