import gym
import numpy as np

def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a

class QLearner(object):
    def __init__(self,
                 num_states,
                 num_actions,
                 alpha=0.2,
                 gamma=0.9,
                 epsilon=0.5,
                 epsilon_decay=0.99,
                 algorithm_type='sarsa'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.algorithm_type = algorithm_type
        # self.q_value = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        self.q_value = np.zeros([num_states, num_actions])

    def set_initial_state(self, curr_s):
        """
        @summary: Sets the initial state and returns an action
        @param curr_s: The initial state
        @returns: The selected action
        """
        self.curr_s = curr_s
        self.curr_a = select_a_with_epsilon_greedy(curr_s, self.q_value, epsilon=self.epsilon)
        return self.curr_a

    def move(self, next_s, r):
        """
        @summary: Moves to the given state with given reward and returns action
        @param next_s: The new state
        @param r: The reward
        @returns: The selected action
        """
        next_a = select_a_with_epsilon_greedy(next_s, self.q_value, epsilon=self.epsilon)

        if self.algorithm_type == 'sarsa':
            delta = r + self.gamma * self.q_value[next_s, next_a] - self.q_value[self.curr_s, self.curr_a]
        elif self.algorithm_type == 'q_learning':
            delta = r + self.gamma * np.max(self.q_value[next_s, :]) - self.q_value[self.curr_s, self.curr_a]
        else:
            raise ValueError("Invalid algorithm_type: {}".format(self.algorithm_type))

        self.q_value[self.curr_s, self.curr_a] += self.alpha * delta
        self.epsilon = self.epsilon * self.epsilon_decay

        self.curr_s = next_s
        self.curr_a = next_a
        return next_a

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy polciy.')
    parser.add_argument('-a', '--algorithm', default='sarsa', choices=['sarsa', 'q_learning'],
                        help="Type of learning algorithm. (Default: sarsa)")
    parser.add_argument('-e', '--environment', default='Roulette-v0',
                        help="Name of the environment provided in the OpenAI Gym. (Default: Roulette-v0)")
    parser.add_argument('-n', '--nepisode', default='20000', type=int,
                        help="Number of episode. (Default: 20000)")
    parser.add_argument('-ms', '--maxstep', default='200', type=int,
                        help="Maximum step allowed in a episode. (Default: 200)")
    args = parser.parse_args()

    env_type = args.environment
    algorithm_type = args.algorithm
    n_episode = args.nepisode
    max_step = args.maxstep

    np.random.RandomState(42)
    env = gym.envs.make(env_type)

    if hasattr(env.observation_space, "n"):
        n_s = env.observation_space.n
    else:
        n_s = env.observation_space.shape[0]
    n_a = env.action_space.n
    learner = QLearner(n_s, n_a, algorithm_type=algorithm_type)

    result_dir = 'results-{0}-{1}'.format(env_type, algorithm_type)
    env.monitor.start(result_dir, force=True)

    for i_episode in xrange(n_episode):
        state = env.reset()
        print state
        action = learner.set_initial_state(state)

        for i_step in xrange(max_step):
            env.render()
            next_state, r, done, info = env.step(action)
            print next_state
            action = learner.move(next_state, r)

            if done:
                break

    env.monitor.close()

if __name__ == "__main__":
    main()
