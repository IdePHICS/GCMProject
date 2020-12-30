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
    def __init__(self, *, task, regularisation, data_model):
        self.task = task
        self.lamb = regularisation
        self.data_model = data_model

    def run_experiment(self, initialisation='uninformed', tolerance=1e-10, damping=0,
                       verbose=False, max_steps=1000, *, sample_complexity):
        '''
        Runs saddle-point equations.
        '''
        self._initialise_model(sample_complexity)

        self.se = StateEvolution(model=self.model,
                       initialisation=initialisation,
                       tolerance=tolerance,
                       damping=damping,
                       verbose=verbose,
                       max_steps=max_steps)

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

        info = self.model.get_info()
        info.update(self.data_model.get_info())
        info.update(self.se.get_info())

        info.update({
            'date': '{}_{}'.format(day,time),
        })

        sub_dir = '{}/{}'.format(data_dir, day)
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)

                    name = '{}/{}_{}.json'.format(sub_dir, info['model'], unique_id)
                    print('Saving experiment at {}'.format(name))
                    with open(name, 'w') as outfile:
                        json.dump(info, outfile)
