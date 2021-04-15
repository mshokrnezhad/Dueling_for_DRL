""" import collections  # used to store collections of data, for example, list, dict, set, tuple etc.
import cv2  # to work with images
import matplotlib.pyplot as plt
import numpy as np
import gym


def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    s_plt1 = fig.add_subplot(111, label="1")  # "111" means "1x1 grid, first subplot",
    # also "234" means "2x3 grid, 4th subplot".
    s_plt2 = fig.add_subplot(111, label="2", frame_on=False)  # "frame_on=False" means showing two subplots
    # in one frame at the same time, transparently

    s_plt1.plot(x, epsilons, color="C0")
    s_plt1.set_xlabel("Training Steps", color="C0")
    s_plt1.set_ylabel("Epsilon", color="C0")
    s_plt1.tick_params(axis="x", color="C0")  # "tick_params" is used to change the appearance of ticks,
    # tick labels, and gridlines.
    s_plt1.tick_params(axis="y", color="C0")

    n = len(scores)
    running_avg = np.empty(n)
    for i in range(n):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    s_plt2.scatter(x, running_avg, color="C1")
    s_plt2.axes.get_xaxis().set_visible(False)  # "axes.get_xaxis()" returns the XAxis instance
    s_plt2.yaxis.tick_right()
    s_plt2.set_ylabel('Score', color="C1")
    s_plt2.yaxis.set_label_position('right')
    s_plt2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):  # Wrappers will allow us to add functionality to environments,
    # such as modifying observations and rewards to be fed to our agent. It is common in reinforcement
    # learning to preprocess observations in order to make them more easy to learn from. A common example
    # is when using image-based inputs, to ensure that all values are between 0 and 1 rather than between
    # 0 and 255, as is more common with RGB images.
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)  # to invoke the constructor of the superclass. env
        # is passed to the super constructor based on the documentation
        self.repeat = repeat
        self.shape = env.observation_space.low.shape  # env.observation_space.low and env.observation_space.high
        # which will print the minimum and maximum values for each observation variable.
        self.frame_buffer = np.zeros_like((2,self.shape))  # np.zeros_like returns an array of zeros with the
        # same shape and type as a given array.
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]  # set reward to 1 if it is > 1 nad -1 if it is < -1
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])  # np.maximum is to take two arrays
        # and compute their element-wise maximum.
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()

        # just to make results comparable to the results of the paper
        no_ops = np.random.randint(self.no_ops+1) if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'Fire'
            obs, _, _, _ = self.env.step(1)
        # just to make results comparable to the results of the paper

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])  # to make the order of image channel as pyTorch expects
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)  # set
        # identical bound for observation_space

    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)  # self.shape[1:]: image
        # size, interpolation=cv2.INTER_AREA: OpenCV's resize() with INTER_AREA only works for images with
        # at most 4 channels when the old image width and height are not an integer multiples of the new
        # width and height (scale factors do not have to be the same for both width and height, as long as
        # both scale factors are integers). For more info, see https://www.programmersought.com/article/11535065011/
        new_observation = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)  # pytorch gets numpy arrays
        new_observation = np.swapaxes(new_observation, 2, 0)
        new_observation = new_observation/255   # normalize the channels

        return new_observation


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0), 
                                                env.observation_space.high.repeat(repeat, axis=0), dtype=np.float32)  # an example of
        # observation_space: Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32). I guess in this
        # line we are trying to shape the observation space regarding the fact that we repeat each action "repeat"
        # times
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_reward=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_reward, no_ops, fire_first)
    env = PreprocessFrame(shape, env)    
    env = StackFrames(env, repeat)

    return env


 """

import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    """ modified from:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
    """
    def __init__(self, env=None, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                              shape=self.shape,dtype=np.float32)
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)

        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = np.swapaxes(new_obs, 2,0)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=n_steps)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs

def make_env(env_name, shape=(84,84,1), skip=4):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, skip)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, skip)

    return env
