from DQNAgent import DQNAgent

class UnityDQNAgent(DQNAgent):
    def __init__(self, env, brain_name, state_size, action_size, **kwargs):
        super().__init__(env, state_size, action_size, **kwargs)
        self.brain_name = brain_name

    def env_reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def env_step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None

    def env_render(self, train_mode=False):
        pass