from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque, defaultdict


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True                      # marks if we're on first time step of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []    # returns of completed episodes in this segment
    ep_lens = []    # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    # XLV: add success percentage
    suc = False
    sucs = np.zeros(horizon, 'int32')

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:

            # XLV: add for reset
            ob = env.reset()
            # XLV: add for deal with 'nan' data
            if len(ep_rets) == 0 and len(ep_lens) == 0:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0

            yield {"ob": obs, "rew": rews, "vpred": vpreds,  "new": news, "suc": sucs,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        # XLV: added for collect success rate
        sucs[i] = suc

        ob, rew, new, info = env.step(ac)
        suc = info['suc']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0

            ob = env.reset()

        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    # XLV: add for computing Mento-Carlo Return
    G = np.append(seg["rew"], 0)
    tdtarget = np.empty(T, 'float32')
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        # XLV: add for update each timestep of G and tdtarget
        G[t] = rew[t] + gamma * G[t+1] * nonterminal
        tdtarget[t] = rew[t] + gamma * vpred[t+1] * nonterminal
    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    # XLV: add for return G except the last element and tdtarget
    seg["mcreturn"] = G[:-1]
    seg["tdtarget"] = tdtarget


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
