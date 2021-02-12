import numpy as np

class StateEvolution(object):
    '''
    Iterator for the saddle-point equations.
    -- args --
    initialisation: initial condition (uninformed or informed)
    tolerante: tolerance for convergence.
    damping: damping coefficient.
    verbose: if true, print step-by-step iteration.
    max_steps: maximum number of steps before convergence
    model: instance of model class. See /models.
    '''
    def __init__(self, initialisation='uninformed', tolerance=1e-10, damping=0,
                 verbose=False, max_steps=1000, *, model):

        self.max_steps = max_steps
        self.init = initialisation
        self.tol = tolerance
        self.damping = damping
        self.model = model
        self.verbose = verbose

        # Status = 0 at initialisation.
        self.status = 0


    def _initialise(self):
        '''
        Initialise saddle-point equations
        '''
        self.overlaps = {
            'variance': np.zeros(self.max_steps+1),
            'self_overlap': np.zeros(self.max_steps+1),
            'teacher_student': np.zeros(self.max_steps+1)
        }

        if self.init == 'uninformed':
            self.overlaps['variance'][0] = 1000
            self.overlaps['self_overlap'][0] = 0.001
            self.overlaps['teacher_student'][0] = 0.001

        elif self.init == 'informed':
            self.overlaps['variance'][0] = 0.001
            self.overlaps['self_overlap'][0] = 0.999
            self.overlaps['teacher_student'][0] = 0.999


    def _get_diff(self, t):
        '''
        Compute differencial between step t+1 and t.
        '''
        diff = np.abs(self.overlaps['variance'][t+1]-self.overlaps['variance'][t])
        diff += np.abs(self.overlaps['self_overlap'][t+1]-self.overlaps['self_overlap'][t])
        diff += np.abs(self.overlaps['teacher_student'][t+1]-self.overlaps['teacher_student'][t])

        return diff

    def damp(self, new, old):
        '''
        Damping function.
        '''
        return (1-self.damping) * new + self.damping * old

    def iterate(self):
        '''
        Iterate the saddle-point equations.
        '''
        self._initialise()

        for t in range(self.max_steps):
            Vtmp, qtmp, mtmp = self.model.update_se(self.overlaps['variance'][t],
                                               self.overlaps['self_overlap'][t],
                                               self.overlaps['teacher_student'][t])

            self.overlaps['variance'][t+1] = self.damp(Vtmp, self.overlaps['variance'][t])
            self.overlaps['self_overlap'][t+1] = self.damp(qtmp, self.overlaps['self_overlap'][t])
            self.overlaps['teacher_student'][t+1] = self.damp(mtmp, self.overlaps['teacher_student'][t])

            diff = self._get_diff(t)

            if self.verbose:
                print('t: {}, diff: {}, self overlaps: {}, teacher-student overlap: {}'.format(t, diff,
                                                                                               self.overlaps['self_overlap'][t+1],
                                                                                               self.overlaps['teacher_student'][t+1]))
            if diff < self.tol:
            # If iterations converge, set status = 1
                if self.verbose:
                    print('Saddle point equations converged with t={} iterations'.format(t+1))

                self.status = 1
                break

        if t == self.max_steps-1:
            # If iterations didn't converge, set status = -1
            if self.verbose:
                print('Saddle point equations did not converge with t={} iterations. Keeping last values'.format(t+1))

            self.status = -1

        self.t_max = t+1

        self.overlaps['variance'] = self.overlaps['variance'][:t+1]
        self.overlaps['self_overlap'] = self.overlaps['self_overlap'][:t+1]
        self.overlaps['teacher_student'] = self.overlaps['teacher_student'][:t+1]

        self.test_error = self.model.get_test_error(self.overlaps['self_overlap'][-1],
                                                    self.overlaps['teacher_student'][-1])

        self.train_loss = self.model.get_train_loss(self.overlaps['variance'][-1],
                                                    self.overlaps['self_overlap'][-1],
                                                    self.overlaps['teacher_student'][-1])

    def get_info(self):
        info = {
            'hyperparameters': {
                'initialisation': self.init,
                'damping': self.damping,
                'max_steps': self.max_steps,
                'tolerance': self.tol
            }
        }
        if self.status != 0:
            info.update({
                'status': self.status,
                'convergence_time': self.t_max,
                'test_error': self.test_error,
                'train_loss': self.train_loss,
                'overlaps': {
                    'variance': self.overlaps['variance'][-1],
                    'self_overlap': self.overlaps['self_overlap'][-1],
                    'teacher_student': self.overlaps['teacher_student'][-1]
                }
            })
        return info
