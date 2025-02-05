import numpy as np
from baselines import logger
from baselines.her.dropout_ensemble_v1 import DropoutEnsemble

FUN_NAME_TO_FUN = {
    'var': lambda vals: np.var(vals, axis=0),
    'std': lambda vals: np.std(vals, axis=0),
    'tanh': lambda vals: np.tanh(np.var(vals, axis=0)),
    'exp': lambda vals: np.exp(np.std(vals, axis=0)),
}

def make_sample_disagreement_goals(static_init_obs, sample_goals_fun, value_ensemble, n_candidates, policy, disagreement_fun_name):
    """
    Disagreement-based goal sampler.
    Args:
        env (baselines.envs.robotics.robot_env.RobotEnv): a temporary env returned from cached_make_env, different from the vector env
        value_ensemble (baselines.her.value_ensemble_v1.ValueEnsemble):
        n_candidates (int): the number of goal candidates sampled to pass in the value ensemble
    """
    disagreement_fun = FUN_NAME_TO_FUN[disagreement_fun_name]

    def _sample_goals(obs_dict, save_path=None):
        o, ag = obs_dict['observation'], obs_dict['achieved_goal']
        n_samples = len(o)
        candidates = sample_goals_fun(n_candidates)
        if static_init_obs:
            # if all initial observations are the same,
            # no need to repeat computation
            o = o[0][np.newaxis, ...]
            ag = ag[0][np.newaxis, ...]
            input_o = np.repeat(o, repeats=n_candidates, axis=0)
            input_ag = np.repeat(ag, repeats=n_candidates, axis=0)
            input_u = None if not value_ensemble.use_Q else policy.get_actions(o=input_o, ag=input_ag, g=candidates)
            vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=candidates,
                                             u=input_u)
            vals = np.squeeze(vals, axis=2)  # (size_ensemble, n_candidates, 1) -> (size_ensemble, n_candidates)

            disagreement = disagreement_fun(vals)
            sum_disagreement = np.sum(disagreement)
            if np.allclose(sum_disagreement, 0):
                disagreement = None
                logger.logkv('ve/stats_p/std', 0)
            else:
                disagreement /= sum_disagreement
                logger.logkv('ve/stats_p/std', np.std(disagreement))

            indices = np.random.choice(n_candidates, size=n_samples, p=disagreement, replace=True)  # FIXME: replace = True or False?
            samples = candidates[indices]
        else:
            input_o = np.repeat(o, repeats=n_candidates, axis=0)
            input_ag = np.repeat(ag, repeats=n_candidates, axis=0)
            input_candidates = np.tile(candidates, reps=(n_samples, 1))
            input_u = None if not value_ensemble.use_Q else policy.get_actions(o=input_o, ag=input_ag, g=input_candidates)
            vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=input_candidates,
                                             u=input_u)
            vals = np.squeeze(vals, axis=2)  # (size_ensemble, n_samples*n_candidates)
            vals = np.reshape(vals, (-1, n_samples, n_candidates))

            disagreement = disagreement_fun(vals) # (n_samples, n_candidates)
            sum_disagreement = np.sum(disagreement, axis=1, keepdims=True)
            indices = []
            for sample_idx in range(n_samples):
                if np.allclose(sum_disagreement[sample_idx], 0):
                    sample_p = None
                    logger.logkv('ve/stats_p/std', 0)
                else:
                    sample_p = disagreement[sample_idx, :] / sum_disagreement[sample_idx]
                    logger.logkv('ve/stats_p/std', np.std(disagreement))
                index = np.random.choice(n_candidates, size=1, p=sample_p)[0]
                indices.append(index)
            samples = candidates[indices]

        # # disabled
        # if save_path:
            # joblib.dump(samples, save_path, compress=3)

        return samples

    return _sample_goals


def make_sample_dummy_goals(sample_goals_fun):
    """
    Sample goals uniformly. This is equivalent to calling env.sample_goals.
    Args:
        sample_goals_fun:

    Returns:

    """
    def _sample_goals(obs_dict, save_path=None):
        return sample_goals_fun(len(obs_dict['observation']))
    return _sample_goals


def make_goal_sampler_factory_random_init_ob(
    sample_goals_fun, value_ensemble, policy, n_candidates, disagreement_fun_name, rnd=None
):
    def goal_sampler(obs_dict):

        if disagreement_fun_name == 'uniform' or value_ensemble.size_ensemble == 0:
            return sample_goals_fun(1)[0]
        elif isinstance(value_ensemble, DropoutEnsemble):
            all_states = sample_goals_fun(n_candidates)
            o = obs_dict['observation'][np.newaxis, ...]
            ag = obs_dict['achieved_goal'][np.newaxis, ...]
            input_o = np.repeat(o, repeats=n_candidates, axis=0)
            input_ag = np.repeat(ag, repeats=n_candidates, axis=0)
            input_u = policy.get_actions(o=input_o, ag=input_ag, g=all_states)
            vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=all_states, u=input_u)

            vals = np.squeeze(vals)  # (size_ensemble, n_candidates, 1) -> (size_ensemble, n_candidates)

            compute_disagreement_fun = FUN_NAME_TO_FUN[disagreement_fun_name]
            disagreement = compute_disagreement_fun(vals)

            sum_disagreement = np.sum(disagreement)
            if np.allclose(sum_disagreement, 0):
                logger.logkv('ve/stats_disag/mean', 0)
                logger.logkv('ve/stats_disag/std', 0)
                disagreement = None
            else:
                logger.logkv('ve/stats_disag/mean', np.mean(disagreement))
                logger.logkv('ve/stats_disag/std', np.std(disagreement))
                disagreement /= sum_disagreement

            return all_states[np.random.choice(np.arange(n_candidates), p=disagreement)]
        elif rnd:
            all_states = sample_goals_fun(n_candidates)
            o = obs_dict['observation'][np.newaxis, ...]
            ag = obs_dict['achieved_goal'][np.newaxis, ...]
            input_o = np.repeat(o, repeats=n_candidates, axis=0)
            input_ag = np.repeat(ag, repeats=n_candidates, axis=0)
            input_u = None if not value_ensemble.use_Q else policy.get_actions(o=input_o, ag=input_ag, g=all_states)
            vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=all_states,
                                             u=input_u)

            vals = np.squeeze(vals, axis=2)  # (size_ensemble, n_candidates, 1) -> (size_ensemble, n_candidates)

            compute_disagreement_fun = FUN_NAME_TO_FUN[disagreement_fun_name]
            disagreement = compute_disagreement_fun(vals)

            rnd_val = rnd.get_values(o=input_o, ag=input_ag, g=all_states)
            rnd_val = np.squeeze(rnd_val)
            rnd_mean = rnd_val.mean()
            rnd_std = rnd_val.std()
            rnd_val = (rnd_val - rnd_mean) / rnd_std
            rnd_val = rnd_val + 1
            rnd_val = np.clip(rnd_val, 0., 1.)
        
            disagreement = disagreement * rnd_val

            sum_disagreement = np.sum(disagreement)

            if np.allclose(sum_disagreement, 0):
                logger.logkv('ve/stats_disag/mean', 0)
                logger.logkv('ve/stats_disag/std', 0)
                disagreement = None
            else:
                logger.logkv('ve/stats_disag/mean', np.mean(disagreement))
                logger.logkv('ve/stats_disag/std', np.std(disagreement))
                disagreement /= sum_disagreement

            return all_states[np.random.choice(np.arange(n_candidates), p=disagreement)]
        else:

            all_states = sample_goals_fun(n_candidates)
            o = obs_dict['observation'][np.newaxis, ...]
            ag = obs_dict['achieved_goal'][np.newaxis, ...]
            input_o = np.repeat(o, repeats=n_candidates, axis=0)
            input_ag = np.repeat(ag, repeats=n_candidates, axis=0)
            input_u = None if not value_ensemble.use_Q else policy.get_actions(o=input_o, ag=input_ag, g=all_states)
            vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=all_states,
                                             u=input_u)

            vals = np.squeeze(vals, axis=2)  # (size_ensemble, n_candidates, 1) -> (size_ensemble, n_candidates)

            compute_disagreement_fun = FUN_NAME_TO_FUN[disagreement_fun_name]
            disagreement = compute_disagreement_fun(vals)

            sum_disagreement = np.sum(disagreement)

            if np.allclose(sum_disagreement, 0):
                logger.logkv('ve/stats_disag/mean', 0)
                logger.logkv('ve/stats_disag/std', 0)
                disagreement = None
            else:
                logger.logkv('ve/stats_disag/mean', np.mean(disagreement))
                logger.logkv('ve/stats_disag/std', np.std(disagreement))
                disagreement /= sum_disagreement

            return all_states[np.random.choice(np.arange(n_candidates), p=disagreement)]

    return goal_sampler