import ray
import time
from copy import deepcopy
#import matplotlib.pyplot as plt
import random
from random import randint, choice
import numpy as np
#%matplotlib inline
import pickle
from numpy.testing import rundocs

import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision = 2)
TransitionProb = [0.7, 0.1, 0.1, 0.1]
def generate_row(length, h_prob):
    row = np.random.choice(2, length, p=[1.0 - h_prob, h_prob])
    row = ''.join(list(map(lambda z: 'F' if z == 0 else 'H', row)))
    return row


def generate_map(shape):
    """

    :param shape: Width x Height
    :return: List of text based map
    """
    h_prob = 0.1
    grid_map = []

    for h in range(shape[1]):

        if h == 0:
            row = 'SF'
            row += generate_row(shape[0] - 2, h_prob)
        elif h == 1:
            row = 'FF'
            row += generate_row(shape[0] - 2, h_prob)

        elif h == shape[1] - 1:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FG'
        elif h == shape[1] - 2:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FF'
        else:
            row = generate_row(shape[0], h_prob)

        grid_map.append(row)
        del row

    return grid_map



MAPS = {

    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "16x16": [
        "SFFFFFFFFHFFFFHF",
        "FFFFFFFFFFFFFHFF",
        "FFFHFFFFHFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFHHFFFFFFFHFFFH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFHFFF",
        "FFFFFHFFFFFFFFFH",
        "FFFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFHFFFFHF",
        "FFFFFFFFFFHFFFFF",
        "FFFHFFFFFFFFFFFG",
    ],

    "32x32": [
        'SFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFH',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFH',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFH',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFH',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFG',
    ]
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        rew_hole = -1000
        rew_goal = 1000
        rew_step = -1

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.TransitProb = np.zeros((nA, nS + 1, nS + 1))
        self.TransitReward = np.zeros((nS + 1, nA))

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'H':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_hole
                    elif letter in b'G':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_goal
                    else:
                        if is_slippery:
                            #for b in [(a-1)%4, a, (a+1)%4]:
                            for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransitionProb):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                #rew = float(newletter == b'G')
                                #li.append((1.0/10.0, newstate, rew, done))
                                if newletter == b'G':
                                    rew = rew_goal
                                elif newletter == b'H':
                                    rew = rew_hole
                                else:
                                    rew = rew_step
                                li.append((p, newstate, rew, done))
                                self.TransitProb[a, s, newstate] += p
                                self.TransitReward[s, a] = rew_step
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def GetSuccessors(self, s, a):
        next_states = np.nonzero(self.TransitProb[a, s, :])
        probs = self.TransitProb[a, s, next_states]
        return [(s,p) for s,p in zip(next_states[0], probs[0])]

    def GetTransitionProb(self, s, a, ns):
        return self.TransitProb[a, s, ns]

    def GetReward(self, s, a):
        return self.TransitReward[s, a]

    def GetStateSpace(self):
        return self.TransitProb.shape[1]

    def GetActionSpace(self):
        return self.TransitProb.shape[0]



def evaluate_policy(env, policy, trials = 1000):
    total_reward = 0
    for _ in range(trials):
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward
    return total_reward / trials

def evaluate_policy_discounted(env, policy, discount_factor, trials = 1000):
    total_reward = 0
    for _ in range(trials):
        env.reset()
        done = False
        observation = 0
        i = 0
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward * discount_factor ** i
            i += 1
    return total_reward / trials

def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np  = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor = beta)))
    #print("State Value image view:\n")
    #plt.imshow(np.array(v_np[:-1]).reshape((map_size,map_size)))

    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))

def sync_value_iteration_v1(env, beta = 0.999, epsilon = 0.0001):

    A = env.GetActionSpace()
    S = env.GetStateSpace()

    pi = [0] * S
    v = [0] * S

    pi_new = [0] * S
    v_new = [0] * S

    bellman_error = float('inf')
    while(bellman_error > epsilon):
        bellman_error = 0
        for state in range(S):
        #    max_v = float('-inf')
        #    max_a = 0
        #    for action in range(A):
            v_new[state], pi_new[state] = max(
                [(env.GetReward(state, action) + \
                 beta * sum(
                    [env.GetTransitionProb(state,action, s) * v[s] for s in range(S)]), action) for action in range(A)],
                    key = lambda x: x[0])
        bellman_error = max([abs(v[state] - v_new[state]) for state in range(S)])

        v = list(v_new)
        pi = list(pi_new)

    return v, pi

def sync_value_iteration_v2(env, beta = 0.999, epsilon = 0.0001):

    A = env.GetActionSpace()
    S = env.GetStateSpace()

    pi = [0] * S
    v = [0] * S

    v_new = [0] * S

    bellman_error = float('inf')
    while(bellman_error > epsilon):
        bellman_error = 0
        for state in range(S):
        #    max_v = float('-inf')
        #    max_a = 0
        #    for action in range(A):
            v_new[state], pi[state] = max(
                [(env.GetReward(state, action) + \
                 beta * sum(
                    [p * v[s] for s, p in env.GetSuccessors(state,action)]), action) for action in range(A)],
                    key = lambda x: x[0])
        bellman_error = max([abs(v[state] - v_new[state]) for state in range(S)])

        v = list(v_new)

    #INSERT YOUR CODE HERE

    return v, pi


@ray.remote
class VI_server_v1(object):
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_index, update_v, update_pi):
        self.v_new[update_index] = update_v
        self.pi[update_index] = update_pi

    def get_error_and_update(self):
        max_error = 0
        for i in range(len(self.v_current)):
            error = abs(self.v_new[i] - self.v_current[i])
            if error > max_error:
                max_error = error
            self.v_current[i] = self.v_new[i]

        return max_error

@ray.remote
def VI_worker_v1(VI_server, data, worker_id, update_state):
        env, workers_num, beta, epsilon = data
        A = env.GetActionSpace()
        S = env.GetStateSpace()

        # get shared variable
        V, _ = ray.get(VI_server.get_value_and_policy.remote())

        # bellman backup

        max_v, max_a = max(
                [(env.GetReward(update_state, action) + \
                 beta * sum(
                    [p * V[s] for s, p in env.GetSuccessors(update_state,action)]), action) for action in range(A)],
                    key = lambda x: x[0])

        ray.get(VI_server.update.remote(update_state, max_v, max_a))

        # return ith worker
        return worker_id

def sync_value_iteration_distributed_v1(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v1.remote(S)
    data_id = ray.put((env, workers_num, beta, epsilon))

    # start the all worker, store their id in a list

    error = float('inf')
    while error > epsilon:
        workers_list = []
        start = 0
        for i in range(workers_num):
            w_id = VI_worker_v1.remote(VI_server, data_id, i, start)
            workers_list.append(w_id)
            start += 1
        for update_state in range(start, S):
            # Wait for one worker finishing, get its reuslt, and delete it from list
            finished_worker_id = ray.wait(workers_list, num_returns = 1, timeout = None)[0][0]
            finish_worker = ray.get(finished_worker_id)
            workers_list.remove(finished_worker_id)

            # start a new worker, and add it to the list
            w_id = VI_worker_v1.remote(VI_server, data_id, finish_worker, update_state)
            workers_list.append(w_id)
        finished_workers = ray.get(workers_list)
        error = ray.get(VI_server.get_error_and_update.remote())

    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi





@ray.remote
class VI_server_v2(object):
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size
        self.errors = []

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_range, update_v, update_pi, update_error):
        update_slice = slice(update_range.start, update_range.stop)
        self.v_new[update_slice] = update_v
        self.pi[update_slice] = update_pi
        self.errors.append(update_error)

    def get_error_and_update(self):
        self.v_current = self.v_new
        max_error = max(self.errors)
        self.errors = []
        return max_error

@ray.remote
def VI_worker_v2(VI_server, data, update_range):
        env, workers_num, beta, epsilon = data
        A = env.GetActionSpace()
        S = env.GetStateSpace()
        V, _ = ray.get(VI_server.get_value_and_policy.remote())
        v_new = []
        p_new = []
        for update_state in update_range:
            max_v, max_a = max(
                [(env.GetReward(update_state, action) + \
                 beta * sum(
                    [p * V[s] for s, p in env.GetSuccessors(update_state,action)]), action) for action in range(A)],
                    key = lambda x: x[0])
            v_new.append(max_v)
            p_new.append(max_a)

        error = max([abs(V[state] - v_new[i]) for i, state in enumerate(update_range)])
        ray.get(VI_server.update.remote(update_range, v_new, p_new, error))

def sync_value_iteration_distributed_v2(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v2.remote(S)
    data_id = ray.put((env, workers_num, beta, epsilon))
    batch_size = int(S / workers_num)

    error = float('inf')
    while error > epsilon:
        workers_list = []
        for update_slice in (range(i, min(S, i+batch_size)) for i in range(0, S, batch_size)):
            workers_list.append(VI_worker_v2.remote(VI_server, data_id, update_slice))
        ray.get(workers_list)
        error = ray.get(VI_server.get_error_and_update.remote())

    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi

map_8 = (MAPS["8x8"], 8)
map_16 = (MAPS["16x16"], 16)
map_32 = (MAPS["32x32"], 32)
#map_50 = (generate_map((50,50)), 50)
#map_110 = (generate_map((110,110)), 110)

MAP = map_32
map_size = MAP[1]
run_time = {}

ray.init(num_cpus=4,include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

beta = 0.999
#print("Game Map:")
#env.render()

@ray.remote
class VI_client_v3(object):
    def __init__(self,update_range):
        self.range = update_range

@ray.remote
class VI_server_v3(object):
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size
        self.errors = []

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_range, update_v, update_pi, update_error):
        update_slice = slice(update_range.start, update_range.stop)
        self.v_new[update_slice] = update_v
        self.pi[update_slice] = update_pi
        self.errors.append(update_error)

    def get_error_and_update(self):
        self.v_current = self.v_new
        max_error = max(self.errors)
        self.errors = []
        return max_error

@ray.remote(num_return_vals=4)
def VI_worker_v3(VI_server, data, update_range):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()
    V, _ = ray.get(VI_server.get_value_and_policy.remote())
    v_new = []
    p_new = []
    for update_state in update_range:
        max_v, max_a = max(
            [(env.GetReward(update_state, action) + \
             beta * sum(
                [p * V[s] for s, p in env.GetSuccessors(update_state,action)]), action) for action in range(A)],
                key = lambda x: x[0])
        v_new.append(max_v)
        p_new.append(max_a)

    error = max([abs(V[state] - v_new[i]) for i, state in enumerate(update_range)])
    #update_id = VI_server.update.remote(update_range, v_new, p_new, error)
    return (update_range, v_new, p_new, error)

def sync_value_iteration_distributed_v3(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v3.remote(S)
    data_id = ray.put((env, workers_num, beta, epsilon))
    batch_size = int((S - 1) / workers_num)

    error = float('inf')
    while error > epsilon:
        workers_list = []
        for update_slice in (range(i, min(S-1, i+batch_size)) for i in range(0, S-1, batch_size)):
            workers_list.append(VI_worker_v3.remote(VI_server, data_id, update_slice))
        #rests = ray.get(workers_list)
        a = ray.get([VI_server.update.remote(r, v, p, e) for r, v, p, e in workers_list])
        error = ray.get(VI_server.get_error_and_update.remote())

    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi



@ray.remote
class VI_client_v4(object):
    def __init__(self,update_range):
        self.range = update_range

@ray.remote
class VI_server_v4(object):
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size
        self.errors = []

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_range, update_v, update_pi, update_error):
        print('updating')
        update_slice = slice(update_range.start, update_range.stop)
        self.v_new[update_slice] = update_v
        self.pi[update_slice] = update_pi
        self.errors.append(update_error)

    def get_error_and_update(self):
        print(self.errors)
        self.v_current = self.v_new
        max_error = max(self.errors)
        self.errors = []
        return max_error

@ray.remote
def VI_worker_v4(VI_server, data, update_range):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()
    V, _ = ray.get(VI_server.get_value_and_policy.remote())
    v_new = []
    p_new = []
    for update_state in update_range:
        max_v, max_a = max(
            [(env.GetReward(update_state, action) + \
             beta * sum(
                [p * V[s] for s, p in env.GetSuccessors(update_state,action)]), action) for action in range(A)],
                key = lambda x: x[0])
        v_new.append(max_v)
        p_new.append(max_a)

    error = max([abs(V[state] - v_new[i]) for i, state in enumerate(update_range)])
    ray.get(VI_server.update.remote(update_range, v_new, p_new, error))
    #return (update_range, v_new, p_new, error)

@ray.remote
def VI_loop(VI_server):
        workers_list = []
        for update_slice in (range(i, min(S-1, i+batch_size)) for i in range(0, S-1, batch_size)):
            workers_list.append(VI_worker_v4.remote(VI_server, data_id, update_slice))
        #rests = ray.get(workers_list)
        a = ray.get([VI_server.update.remote(r, v, p, e) for r, v, p, e in workers_list])
        error = ray.get(VI_server.get_error_and_update.remote())

def sync_value_iteration_distributed_v4(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v4.remote(S)
    data_id = ray.put((env, workers_num, beta, epsilon))
    batch_size = int((S - 1) / workers_num)

    error = float('inf')
    workers_list = []
    for i in range(workers_num):
        workers_list.append(VI_worker_v4.remote(VI_server, data_id, range(S-1)))
        #VI_server.update.remote(*worker_list[-1])

    while error > epsilon:
        finished_worker_id = ray.wait(workers_list, num_returns = 1, timeout = None)[0][0]
        ray.get(finished_worker_id)
        workers_list.remove(finished_worker_id)

        # start a new worker, and add it to the list
        w_id = VI_worker_v4.remote(VI_server, data_id, range(S-1))
        workers_list.append(w_id)
        #for update_slice in (range(i, min(S-1, i+batch_size)) for i in range(0, S-1, batch_size)):
        #workers_list.append(VI_worker_v4.remote(VI_server, data_id, range(S-1)))
        #rests = ray.get(workers_list)
        error = ray.get(VI_server.get_error_and_update.remote())

    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi


def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4):
    pass


def run_test(func, **kwargs):
    times = []
    pis = []
    for the_map in [map_8, map_16, map_32]:
        MAP = the_map
        map_size = MAP[1]
        env = FrozenLakeEnv(desc = MAP[0], is_slippery = True)
        start_time = time.time()
        v, pi = func(env, **kwargs)
        #v, pi = sync_value_iteration_v2(env, beta = beta)
        end_time = time.time()
        times.append(end_time - start_time)
        v_np, pi_np  = np.array(v), np.array(pi)
        pis.append(pi_np)
        run_time['Sync distributed v2'] = end_time - start_time
        print("time:", run_time['Sync distributed v2'])
        print_results(v, pi, map_size, env, beta, 'dist_vi_v2')
    return times, pis

out = ""
#sv1_time, sv1_policy = run_test(sync_value_iteration_v1, beta = beta)
#print("Sync v1", sv1_time)
#out += f'Sync v1 {sv1_time}\n'
sv2_time, sv2_policy = run_test(sync_value_iteration_v2, beta = beta)
print("Sync v2", sv2_time)
out += f'Sync v2 {sv2_time}\n'
for workers in [2, 4, 8]:
    #dv1_time, policy = run_test(sync_value_iteration_distributed_v1, beta = beta, workers_num = workers)
    #print("Dist-Sync v1:", workers, "workers", dv1_time)
    #out += f'Dist-Sync v1: {workers} workers {dv1_time}\n'
    #dv2_time, policy = run_test(sync_value_iteration_distributed_v2, beta = beta, workers_num = workers)
    #print("Dist-Sync v2:", workers, "workers", dv2_time)
    #out += f'Dist-Sync v2: {workers} workers {dv2_time}\n'
    #if not np.prod(policy == sv2_policy):
    #    print('POLICIES DO NOT MATCH!!')
    #dv3_time, policy = run_test(sync_value_iteration_distributed_v3, beta = beta, workers_num = workers)
    #print("Dist-Sync v3", workers, "workers", dv3_time)
    #out += f'Dist-Sync v3: {workers} workers {dv3_time}\n'
    #if not np.prod(policy == sv2_policy):
    #    print('POLICIES DO NOT MATCH!!')
    dv4_time, policy = run_test(sync_value_iteration_distributed_v4, beta = beta, workers_num = workers)
    print("Dist-Sync v4", workers, "workers", dv4_time)
    out += f'Dist-Sync v4: {workers} workers {dv4_time}\n'
    if not np.prod(policy == sv2_policy):
        print('POLICIES DO NOT MATCH!!')

print(out)
