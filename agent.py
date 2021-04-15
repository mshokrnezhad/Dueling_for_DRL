import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer


class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,  mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                  replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min       
        self.eps_dec = eps_dec
        self.replace_target_counter = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+"_q_eval",
                                   checkpoint_dir=self.checkpoint_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+"_q_next",
                                   checkpoint_dir=self.checkpoint_dir)

    def store_transition(self, state, action, reward, resulted_state, done):
        self.memory.store_transition(
            state, action, reward, resulted_state, done)

    def sample_memory(self):
        state, action, reward, resulted_state, done = self.memory.sample_buffer(
            self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        resulted_states = T.tensor(resulted_state).to(self.q_eval.device)

        return states, actions, rewards, resulted_states, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(
                self.q_eval.device)  # converting observation to tensor,
            # and observation is in the list because our convolution expects an input tensor of shape batch size
            # by input dims.
            _, advantages = self.q_eval.forward(state)
            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_counter is not None and \
            self.learn_step_counter % self.replace_target_counter == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, resulted_states, dones = self.sample_memory()

        indexes = np.arange(self.batch_size)





        V_states, A_states = self.q_eval.forward(states)
        q_pred = T.add(V_states, (A_states - A_states.mean(dim=1, keepdim=True)))[indexes, actions]

        V_resulted_states, A_resulted_states = self.q_next.forward(resulted_states)
        q_next = T.add(V_resulted_states, (A_resulted_states - A_resulted_states.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0

        target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()