import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)
        self.resulted_state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, resulted_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.resulted_state_memory[index] = resulted_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done       
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)  # "replace=False" assures that no repetitive memory is selected in batch
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        resulted_states = self.resulted_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, resulted_states, terminal
