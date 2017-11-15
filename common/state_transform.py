from __future__ import division
import numpy as np
from collections import OrderedDict


def get_state_names(all=False, obst=False):
    names = ['pelvis_' + n for n in ('rot', 'x', 'y')]
    names += ['pelvis_vel_' + n for n in ('rot', 'x', 'y')]
    names += ['hip_right', 'knee_right', 'ankle_right', 'hip_left', 'knee_left', 'ankle_left']
    names += ['hip_right_vel', 'knee_right_vel', 'ankle_right_vel', 'hip_left_vel', 'knee_left_vel', 'ankle_left_vel']
    names += ['mass_x', 'mass_y']
    names += ['mass_x_vel', 'mass_y_vel']

    if all:
        names += [b + '_' + i for b in ['head', 'pelvis2', 'torso', 'toes_left',
                                        'toes_right', 'talus_left', 'talus_right'] for i in
                  ['x', 'y']]
    else:
        names += [b + '_' + i for b in ['head', 'torso', 'toes_left', 'toes_right',
                                        'talus_left', 'talus_right'] for i in
                  ['x', 'y']]

    names += ['muscle_left', 'muscle_right']
    if obst:
        names += ['obst_dist', 'obst_y', 'obst_r']
    return names


def get_names_to_center(centr):
    if centr == 'pelvis':
        pelvis_or_mass = 'mass'
    elif centr == 'mass':
        pelvis_or_mass = 'pelvis'
    else:
        raise ValueError('centr should be in [mass or pelvis], not {}'.format(centr))
    return [b + '_x' for b in ['head', pelvis_or_mass, 'torso', 'toes_left',
                               'toes_right', 'talus_left', 'talus_right']]


def get_bodies_names():
    return [b + '_' + i for b in ['head', 'torso', 'toes_left', 'toes_right', 'talus_left', 'talus_right']
            for i in ['x', 'y']]


def get_names_obstacles():
    return ['toes_left', 'toes_right', 'talus_left', 'talus_right']


def calculate_velocity(cur, prev):
    if prev is None:
        return np.zeros_like(cur)
    return 100.*(cur - prev)


def _get_pattern_idxs(lst, pattern):
    idxs = [i for i, x in enumerate(lst) if pattern in x]
    return idxs


class State(object):
    def __init__(self, obstacles_mode='bodies_dist', obst_grid_dist=1,
                     grid_points=100, predict_bodies=True, add_step=True, osb_first=False):
        assert obstacles_mode in ['exclude', 'grid', 'bodies_dist', 'standard']

        self.state_idxs = [i for i, n in enumerate(get_state_names(True, True)) if n not in ['pelvis2_x', 'pelvis2_y']]
        self.state_names = get_state_names()
        self.step = 0
        self.add_step = add_step
        self.osb_first = osb_first
        self.obstacles_mode = obstacles_mode
        self.obstacles = OrderedDict()

        self.obst_names = []
        if obstacles_mode == 'standard':
            self.obst_names = ['obst_dist', 'obst_y', 'obst_r']
        elif obstacles_mode == 'grid':
            self.obst_names = ['obst_grid_{}'.format(i) for i in range(grid_points)]
            self.obst_grid_dist = obst_grid_dist
            self.obst_grid_points = grid_points
            self.obst_grid_size = obst_grid_dist * 2 / grid_points
        elif obstacles_mode == 'bodies_dist':
            self._obst_names = get_names_obstacles()
            for i in range(3):
                for n in self._obst_names:
                    self.obst_names.append('{}_{}_obst_x_start'.format(n, i))
                    self.obst_names.append('{}_{}_obst_x_end'.format(n, i))
                    self.obst_names.append('{}_{}_obst_y'.format(n, i))
        self.obst_names.append('is_obstacle')

        if self.add_step:
            self.state_names.append('step')

        self.predict_bodies = predict_bodies
        self.bodies_idxs_x = [self.state_names.index(n) for n in get_bodies_names() if n.endswith('_x')]
        self.bodies_idxs_y = [self.state_names.index(n) for n in get_bodies_names() if n.endswith('_y')]
        self.bodies_idxs = self.bodies_idxs_x + self.bodies_idxs_y
        self.mass_x_idx = self.state_names.index('mass_x')
        self.mass_y_idx = self.state_names.index('mass_y')

        self.state_names_out = self.state_names
        self._set_left_right()

    def _set_left_right(self):
        self.left_idxs = _get_pattern_idxs(self.state_names, '_left')
        self.right_idxs = _get_pattern_idxs(self.state_names, '_right')

    def reset(self):
        self.step = 0
        self.prev_orig = None
        self.prev_pred = None
        self.obstacles = OrderedDict()

    def _predict_bodies(self, state):
        state = np.copy(state)

        if self.step > 0:

            def update_bodies(cur, prev_orig, prev_pred, d):
                flt = cur == prev_orig
                cur[flt] = prev_pred[flt] + d

            # does not matter orig or pred
            dx = state[self.mass_x_idx] - self.prev_orig[self.mass_x_idx]
            dy = state[self.mass_y_idx] - self.prev_orig[self.mass_y_idx]

            cur_bodies_x = state[self.bodies_idxs_x]
            cur_bodies_y = state[self.bodies_idxs_y]

            # need for filter
            prev_orig_bodies_x = self.prev_orig[self.bodies_idxs_x]
            prev_orig_bodies_y = self.prev_orig[self.bodies_idxs_y]

            # need for updating
            prev_pred_bodies_x = self.prev_pred[self.bodies_idxs_x]
            prev_pred_bodies_y = self.prev_pred[self.bodies_idxs_y]

            update_bodies(cur_bodies_x, prev_orig_bodies_x, prev_pred_bodies_x, dx)
            update_bodies(cur_bodies_y, prev_orig_bodies_y, prev_pred_bodies_y, dy)

            state[self.bodies_idxs_x] = cur_bodies_x
            state[self.bodies_idxs_y] = cur_bodies_y
        return state

    def _add_obstacle(self, state):
        pelvis_x = state[1]
        obstacle_x = state[-3]

        if obstacle_x != 100:
            obstacle_x += pelvis_x
            if round(obstacle_x, 5) not in self.obstacles:
                self.obstacles[round(obstacle_x, 5)] = [obstacle_x, state[-2], state[-1]]
                #print('obstacles {}, step {}'.format(self.obstacles.keys(), self.step))
        if len(self.obstacles) > 3:
            Warning('more than 3 obstacles')

    def _get_obstacle_state_reward(self, state):
        is_obst = float(state[-3] != 100)

        if self.obstacles_mode == 'exclude':
            return [is_obst], 0.
        elif self.obstacles_mode == 'standard':
            if not is_obst:
                return [-1., 0., 0., is_obst], 0.
            obst_features = np.clip(state[-3:], -10., 10.)
            return np.append(obst_features, is_obst), 0.
        elif self.obstacles_mode == 'gird':
            mass_x = state[self.state_names.index('mass_x')]
            obst_grid = np.zeros(self.obst_grid_points)
            for k, v in self.obstacles.iteritems():
                obst_x, obst_y, obst_r = v
                obst_h = obst_y + obst_r
                obst_left = int(np.ceil((obst_x - mass_x - obst_r) / self.obst_grid_size) + self.obst_grid_points // 2)
                obst_right = int(np.ceil((obst_x - mass_x + obst_r) / self.obst_grid_size) + self.obst_grid_points // 2)
                obst_left = max(obst_left, 0)
                obst_right = max(obst_right, -1)
                obst_grid[obst_left:obst_right + 1] = obst_h
            obst_features = np.append(obst_grid, is_obst)
            return obst_features, 0
        else:
            obst_state = []
            obst_reward = 0
            for i in range(3):
                if i >= len(self.obstacles):
                    for n in self._obst_names:
                        body_y = state[self.state_names.index(n + '_y')]
                        obst_state.extend([10, 10, body_y])
                else:
                    v = self.obstacles.values()[i]
                    obst_x, obst_y, obst_r = v
                    obst_h = obst_y + obst_r
                    obst_x_start = obst_x - obst_r
                    obst_x_end = obst_x + obst_r
                    for n in self._obst_names:
                        body_x = state[self.state_names.index(n + '_x')]
                        body_y = state[self.state_names.index(n + '_y')]
                        obst_state.append(obst_x_start - body_x)
                        obst_state.append(obst_x_end - body_x)
                        obst_state.append(body_y - obst_h)
                        if obst_reward >= 0 and body_x >= (obst_x_start - obst_r/2) \
                                and (body_x <= obst_x_end+obst_r/2) and (obst_h + obst_r/2) >= body_y:
                            obst_reward = -0.5
            obst_state.append(is_obst)
            return np.asarray(obst_state), obst_reward

    def process(self, state):
        state = np.asarray(state)
        state = state[self.state_idxs]

        if self.osb_first and self.step == 0:
            state[-3:] = [100, 0, 0]

        self._add_obstacle(state)
        obst_state, obst_reward = self._get_obstacle_state_reward(state)
        state_orig = state[:-3]

        if self.add_step:
            state_orig = np.append(state_orig, 1. * self.step / 1000)

        if self.predict_bodies:
            state = self._predict_bodies(state_orig)
        else:
            state = state_orig

        self.step += 1
        self.prev_orig = state_orig
        self.prev_pred = np.copy(state)

        return (state, obst_state), obst_reward

    def flip_state(self, state, copy=True):
        assert np.ndim(state) == 1
        state = np.asarray(state)
        state = self.flip_states(state.reshape(1, -1), copy)
        return state.ravel()

    def flip_states(self, states, copy=True):
        assert np.ndim(states) == 2
        states = np.asarray(states)
        if copy:
            states = states.copy()
        left = states[:, self.left_idxs]
        right = states[:, self.right_idxs]
        states[:, self.left_idxs] = right
        states[:, self.right_idxs] = left
        return states

    @property
    def state_size(self):
        return len(self.state_names_out) + len(self.obst_names)


class StateVel(State):
    def __init__(self, vel_states=get_bodies_names(), obstacles_mode='bodies_dist',
                 add_step=True, predict_bodies=True, osb_first=False):
        super(StateVel, self).__init__(obstacles_mode=obstacles_mode,
                                       predict_bodies=predict_bodies,
                                       add_step=add_step,
                                       osb_first=osb_first)
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.prev_vals = None
        self.state_names += [n + '_vel' for n in vel_states]
        self.state_names_out = self.state_names
        # left right idxs
        self._set_left_right()

    def reset(self):
        super(StateVel, self).reset()
        self.prev_vals = None

    def process(self, state):
        (state, obst_state), obst_reward = super(StateVel, self).process(state)
        cur_vals = state[self.vel_idxs]
        vel = calculate_velocity(cur_vals, self.prev_vals)
        self.prev_vals = cur_vals
        state = np.concatenate((state, vel, obst_state))
        return state, obst_reward


class StateVelCentr(State):
    def __init__(self, centr_state='pelvis_x', vel_states=get_bodies_names(),
                 states_to_center=get_names_to_center('pelvis'),
                 vel_before_centr=True, obstacles_mode='bodies_dist',
                 exclude_centr=False, predict_bodies=True,
                 add_step=True, osb_first=False):
        super(StateVelCentr, self).__init__(obstacles_mode=obstacles_mode,
                                            predict_bodies=predict_bodies,
                                            add_step=add_step,
                                                osb_first=osb_first)

        # center
        self.centr_idx = self.state_names.index(centr_state)
        self.states_to_center = [self.state_names.index(k) for k in states_to_center]
        # velocities
        self.prev_vals = None
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.vel_before_centr = vel_before_centr
        self.state_names += [n + '_vel' for n in vel_states]
        self.exclude_centr = exclude_centr

        if self.exclude_centr:
            self.state_names_out = self.state_names[:max(0, self.centr_idx)] + \
                          self.state_names[self.centr_idx + 1:]
        else:
            self.state_names_out = self.state_names

        # left right idxs
        self._set_left_right()

    def _set_left_right(self):
        state_names = self.state_names_out
        self.left_idxs = _get_pattern_idxs(state_names, '_left')
        self.right_idxs = _get_pattern_idxs(state_names, '_right')

    def reset(self):
        super(StateVelCentr, self).reset()
        self.prev_vals = None

    def process(self, state):
        (state, obst_state), obst_reward = super(StateVelCentr, self).process(state)

        if self.vel_before_centr:
            cur_vals = state[self.vel_idxs]
            vel = calculate_velocity(cur_vals, self.prev_vals)
            self.prev_vals = cur_vals
            state[self.states_to_center] -= state[self.centr_idx]
        else:
            state[self.states_to_center] -= state[self.centr_idx]
            cur_vals = state[self.vel_idxs]
            vel = calculate_velocity(cur_vals, self.prev_vals)
            self.prev_vals = cur_vals

        if self.exclude_centr:
            state = np.concatenate([state[:max(0, self.centr_idx)], state[self.centr_idx+1:]])

        state = np.concatenate((state, vel, obst_state))
        return state, obst_reward
