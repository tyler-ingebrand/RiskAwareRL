from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ra_d4pg_stable.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for ra_d4pg agents.
    """

    def __init__(self, obs_dim, act_dim, size, n_step_lookahead, gamma):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.n_step_lookahead = n_step_lookahead
        self.gamma = gamma
        self.gamma_buf = np.zeros(size, dtype=np.float32)

        # this will store transitions until we reach our n_step_lookahead. Then, it will keep the last n steps
        # so we can calculate n step return. When an episode terminates, we add all remaining experiences to the experience
        # replay buffer
        self.intermediate_buffer = []

    def add_experience(self, obs, act, rew, next_obs, done, gamma):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.gamma_buf[self.ptr] = gamma
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def store(self, obs, act, rew, next_obs, done, timed_out):
        # add tuple to intermediate buffer
        self.intermediate_buffer.append((obs, act, rew, next_obs, done))

        # if intermediate buffer has N steps in it, then compute n step return and add to experience replay
        if len(self.intermediate_buffer) >= self.n_step_lookahead:
            obs, act, rew, next_obs, done = self.intermediate_buffer.pop(0)  # retrieves and removes
            gamma = self.gamma
            for index in range(self.n_step_lookahead-1):
                _, _, temp_rew, next_obs, done = self.intermediate_buffer[index]
                rew += gamma * temp_rew
                gamma *= self.gamma
            self.add_experience(obs, act, rew, next_obs, done, gamma)


        # if end of episode, need to add all remaining experiences to experience replay
        # This is because a n step return with a transition discontinuity (IE reset) does not make sense.
        if done or timed_out:
            while len(self.intermediate_buffer) > 0:
                obs, act, rew, next_obs, done = self.intermediate_buffer.pop(0)  # retrieves and removes
                gamma = self.gamma
                for index in range(min(self.n_step_lookahead, len(self.intermediate_buffer))):
                    _, _, temp_rew, next_obs, done = self.intermediate_buffer[index]
                    rew += gamma * temp_rew
                    gamma *= self.gamma
                self.add_experience(obs, act, rew, next_obs, done, gamma)
            assert len(self.intermediate_buffer) == 0


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     gamma=self.gamma_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def ra_d4pg_stable(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.999, pi_lr=0.0005, q_lr=0.0005, batch_size=256, start_steps=10000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1,
         number_atoms=51, v_min=-1.0, v_max=1.0, n_step_lookahead=1,
         max_alpha=0.5, initial_alpha=-1.0):
    """
    Distributed Distributional Deep Deterministic Policy Gradient (ra_d4pg)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to ra_d4pg.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        number_atoms (int): How many bins to divide the value distribution into.
            We assume a discretized value distribution because it is flexible and
            computationally friendly.

        v_max (float): The maximum possible value. Forms an upper bound on our discrete
            distribution

        v_min (float): The minimum possible value. Forms a lower bound on our discrete
            distribution

        n_step_lookahead (int): How far ahead to look for TD updates. Defaults to 1. If greater than 1, technically
            makes this on-policy, since it does not store intermediate actions and assumes the intermediate reward is optimal.

        max_alpha (float): How to balance expected value and standard deviation. The agent will try to maximize
            EV - alpha * standard deviation. For exploration purposes, alpha starts at a negative value (encouraging
            actions with high variance as this means we don't understand the actions value) and increases until it is
            positive. This is the largest value alpha will be.

        initial_alpha (float): The initial alpha value. This should be negative to encourage exploration. Alpha will
            increase from initial_alpha to max_alpha over the course of training.


    """
    assert n_step_lookahead >= 1 and v_max > v_min and number_atoms > 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn() # seed 12345 for testing
    env.seed(seed), test_env.seed(12345)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Each bin corresponds to the probability that the value is between that bins value in the value vector and the next
    # so the first bin represents the probability that the value is between v_min and v_min+(v_max-vmin)/number_atoms
    value_vector = torch.linspace(v_min, v_max, steps=number_atoms, requires_grad=False)
    value_bin_width = (v_max-v_min)/(number_atoms-1)

    # values for risk aware d4pg
    alpha = initial_alpha

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, number_atoms, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, n_step_lookahead=n_step_lookahead, gamma=gamma)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)


    # Set up function for computing ra_d4pg Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d, local_gamma = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['gamma'].reshape(-1 , 1)


        raw_q = ac.q(o,a)
        probabilities = torch.softmax(raw_q, dim=1)
        expected_value = torch.sum(value_vector * probabilities, dim=1)


        # Bellman backup for Q function
        with torch.no_grad():

            next_raw_q = ac_targ.q(o2, ac_targ.pi(o2))
            next_probabilities = torch.softmax(next_raw_q, dim=1)

            # this is the value distribution at the next state, for all the batches
            # result is B x A, where B = number batches and A = number atoms
            next_discounted_values = torch.reshape(local_gamma * value_vector, (batch_size ,number_atoms) )
            next_not_done = torch.reshape(1-d, (batch_size,1))
            next_value_vector = torch.add(r.reshape(batch_size, 1), next_not_done * next_discounted_values)

            # clamp to be within bounds of our v_min and v_max
            clamped_next_value_vector = torch.clamp(next_value_vector, v_min, v_max)
            # For each next value, subtract current value vector to see how related they are.
            # after this operation, should be B x A x A
            tiled_values = clamped_next_value_vector.repeat(1,1,number_atoms).reshape(batch_size, number_atoms, number_atoms) # is now number_atoms X number_atoms

            #  Take absolute value, and divide by width of bin. This is how many bin widths apart they are.
            tiled_values = torch.abs(tiled_values - value_vector.reshape(number_atoms, 1)) / value_bin_width

            # take 1 - how many bins apart they are, and clamp to 0. Thus, if the value is 1, the bins line up perfectly.
            # if the value is 0, the bins are more than 1 bin width apart. for a value in between, they partially overlap
            tiled_values = torch.clamp(1.0 - tiled_values, 0.0, 1.0)

            # multiply py probability at next state to get a projected distribution
            projected_probabilities = torch.matmul(tiled_values, next_probabilities.reshape(batch_size, number_atoms, 1)).reshape(batch_size, number_atoms)

        # Entropy based loss between distributions
        loss_q = torch.nn.BCELoss(reduction="none")(probabilities, projected_probabilities).mean()

        # Useful info for logging
        loss_info = dict(QVals=expected_value.detach().numpy())
        return loss_q, loss_info

    # Set up function for computing ra_d4pg pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_raw = ac_targ.q(o, ac.pi(o))
        probabilities = torch.softmax(q_raw, dim=1)
        expected_value = torch.sum(probabilities * value_vector, dim=1)
        variance = torch.sum( (expected_value.reshape(-1, 1) - value_vector.reshape(1, -1))**2 * probabilities , dim=1)
        standard_deviation = torch.where(variance > 0, torch.sqrt(variance), torch.tensor(0.)) # remove posibility of sqrt(0.0) as the derivative is inf, which causes NaN
        # note we perform gradient descent, so Min -(EV-alpha*SD) = Max EV-alpha*SD
        return - (expected_value.mean() - alpha * standard_deviation.mean())

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next ra_d4pg step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # linearly goes from initial_alpha to max_alpha in half the total timesteps, then is constant
        alpha = min(initial_alpha + t * (max_alpha-initial_alpha)/(total_steps/2),
                    max_alpha)

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        timed_out = ep_len==max_ep_len
        d = False if timed_out else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, timed_out)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Alpha', alpha)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ra_d4pg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ra_d4pg_stable(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
