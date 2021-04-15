import numpy as np
from agent import Agent
from utils import plot_learning_curve, make_env
import torch as T
from gym import wrappers

env = make_env('PongNoFrameskip-v4')
best_score = -np.inf
load_checkpoint = True
n_games = 1
agent = Agent(gamma=0.99, epsilon=0.1, lr=0.0001, input_dims=(env.observation_space.shape),
              n_actions=(env.action_space.n), mem_size=50000, eps_min=0.1, batch_size=32,
              replace=1000, eps_dec=1e-5, checkpoint_dir='models/', algo='DDQNAgent',
              env_name='PongNoFrameskip-v4')

agent.load_models()
print(agent.q_eval)

#env = wrappers.Monitor(env, "tmp/dqn-video", video_callable=lambda episode_id: True, force=True)

n_steps = 0
score = 0
done = False
obs = env.reset()

while not done:
    action = agent.choose_action(obs)
    resulted_obs, reward, done, info = env.step(action)
    score += reward
    obs = resulted_obs
    n_steps += 1
    print(n_steps)


print('score:', score)
