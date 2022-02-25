import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, ReLU, Linear, Layer
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import math
import numpy as np
import subprocess as sp
import random
import time
from pyglet import clock
from nes_py._image_viewer import ImageViewer
from pynput.keyboard import Listener
 
record_key = []
 
def on_press(key):
    global record_key
    key = str(key)[1]
    if key in ('a', 'A', 'd', 'D', 'o', 'O'):
        record_key.append(ord(key))
        record_key = list(set(record_key))
        if len(record_key) > 2:
            record_key.pop(0)


def on_release(key):
    global record_key
    key = str(key)[1]
    if key in ('a', 'A', 'd', 'D', 'o', 'O'):
        try:
            record_key.remove(ord(key))
        except:
            pass

# the sentinel value for "No Operation"
_NOP = 0

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

        self.img = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # if self.monitor:
        #     self.monitor.record(state)
        self.img = state
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.img = self.env.reset()
        return process_frame(self.img)


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env

def conv_out(In):
        return (In-3+2*1)//2+1
        # (input−kernel_size+2*padding)//stride+1

class MARIO(Layer):
    def __init__(self, actions, obs_dim):
        super(MARIO, self).__init__()
        self.channels = 32
        self.kernel = 3
        self.stride = 2
        self.padding = 1
        self.fc = self.channels*math.pow(conv_out(conv_out(conv_out(conv_out(obs_dim[-1])))),2)
        self.conv0 = Conv2D(out_channels=self.channels, 
                                    kernel_size=self.kernel, 
                                    stride=self.stride, 
                                    padding=self.padding, 
                                    dilation=[1, 1], 
                                    groups=1, 
                                    in_channels=obs_dim[1])
        self.relu0 = ReLU()
        self.conv1 = Conv2D(out_channels=self.channels, 
                                    kernel_size=self.kernel, 
                                    stride=self.stride, 
                                    padding=self.padding, 
                                    dilation=[1, 1], 
                                    groups=1, 
                                    in_channels=self.channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels=self.channels, 
                                    kernel_size=self.kernel, 
                                    stride=self.stride, 
                                    padding=self.padding, 
                                    dilation=[1, 1], 
                                    groups=1, 
                                    in_channels=self.channels)
        self.relu2 = ReLU()
        self.conv3 = Conv2D(out_channels=self.channels, 
                                    kernel_size=self.kernel, 
                                    stride=self.stride, 
                                    padding=self.padding, 
                                    dilation=[1, 1], 
                                    groups=1, 
                                    in_channels=self.channels)
        self.relu3 = ReLU()
        self.linear0 = Linear(in_features=int(self.fc), out_features=512)
        self.linear1 = Linear(in_features=512, out_features=actions)
        self.linear2 = Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = paddle.to_tensor(data=x)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = paddle.reshape(x, shape=[1, -1])
        x = self.linear0(x)
        logits = self.linear1(x)
        value = self.linear2(x)
        return logits, value

def main(world, stage, callback=None, listener=None):
    actions = SIMPLE_MOVEMENT
    obs_dim = [1, 4, 84, 84]
    env = create_train_env(world, stage, actions,"./video/mario_{}_{}.avi".format(world, stage))
    env_human = create_train_env(world, stage, actions,"./video/mario_{}_{}.avi".format(world, stage))
    # get the mapping of keyboard keys to actions in the environment
    if hasattr(env_human, 'get_keys_to_action'):
        keys_to_action = env_human.get_keys_to_action()
    elif hasattr(env_human.unwrapped, 'get_keys_to_action'):
        keys_to_action = env_human.unwrapped.get_keys_to_action()
    else:
        raise ValueError('env has no get_keys_to_action method')

    # create the image viewer
    viewer = ImageViewer(
        env_human.spec.id if env_human.spec is not None else env_human.__class__.__name__,
        env_human.observation_space.shape[0], # height
        env_human.observation_space.shape[1], # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
    )

    # prepare frame rate limiting
    target_frame_duration = 1 / env_human.metadata['video.frames_per_second']
    last_frame_time = 0
    paddle.disable_static()
    params = paddle.load('./models/mario_{}_{}.pdparams'.format(world, stage))
    model = MARIO(len(actions), obs_dim)
    model.set_dict(params)
    model.eval()
    state = env.reset()
    state_human = env_human.reset()
    env_human.img = cv2.cvtColor(env_human.unwrapped.screen, cv2.COLOR_BGR2RGB)
    start_img = cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
    start_img_0 = cv2.putText(start_img, "READY?", (60,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (255, 255, 255), 5)
    cv2.imshow('mario challenge', start_img_0)
    cv2.waitKey(1000)
    start_img = cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
    start_img_3 = cv2.putText(start_img, "3", (380,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 15, (255, 255, 255), 5)
    cv2.imshow('mario challenge', start_img_3)
    cv2.waitKey(1000)
    start_img = cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
    start_img_2 = cv2.putText(start_img, "2", (380,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 15, (255, 255, 255), 5)
    cv2.imshow('mario challenge', start_img_2)
    cv2.waitKey(1000)
    start_img = cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
    start_img_1 = cv2.putText(start_img, "1", (380,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 15, (255, 255, 255), 5)
    cv2.imshow('mario challenge', start_img_1)
    cv2.waitKey(1000)

    human_reward = 0
    agent_reward = 0

    while True:
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            continue
        # save frame beginning time for next refresh
        last_frame_time = current_frame_time
        # clock tick
        clock.tick()
        # reset if the environment is done
        logits, value = model(state)
        policy = F.softmax(logits).numpy()
        action = np.argmax(policy)
        state, reward, done, info = env.step(action)
        state = np.array(state).astype('float32')
        action = keys_to_action.get(tuple(record_key), _NOP)
        _s, _r, _d, _i = env_human.step(action)

        human_reward += _r
        agent_reward += reward

        env_human.img = cv2.cvtColor(env_human.unwrapped.screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('mario challenge', cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512)))
        cv2.waitKey(15)
        # viewer.show(env.unwrapped.screen)
        # pass the observation data through the callback
        if callback is not None:
            callback(_s, _r, _d, _i)
        # shutdown if the escape key is pressed
        if viewer.is_escape_pressed:
            break

        if done or _d:
            if human_reward >= agent_reward:
                end_img = cv2.putText(cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
                                    , "WIN!!!", (170,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (255, 255, 255), 5)
            else:
                end_img = cv2.putText(cv2.resize(np.concatenate([env_human.img, env.img], axis=1), (960, 512))
                                    , "LOSE~~", (40,256), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (255, 255, 255), 5)
            cv2.imshow('mario challenge', end_img)
            cv2.waitKey(2000)
            done, _d = False, False
            break

    viewer.close()
    env.close()
    env_human.close()

if __name__ == "__main__":
    with Listener(on_press = on_press, on_release = on_release) as listener:
        cv2.namedWindow('mario challenge', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mario challenge', (960, 512))
        env_list = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3)]
        while True:
            world , stage = random.choice(env_list)
            main(world, stage, listener = listener)
        listener.join()