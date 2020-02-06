from pongEnv import Pala

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import adam
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2


env = Pala()
np.random.seed(0)
tensor_board = TensorBoard(log_dir='./logs')  # Para crear logs


class PPO:
    """ Implementacion del algoritmo PPO"""

    def __init__(self, action_space, state_space):
        self.epsilon = 1
        self.gamma = .95
        self.clipping_val = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.001
        self.action_space = action_space
        self.state_space = state_space
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))  # Aqui va el custom
        return model

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        gamma = 0.99
        lmbda = 0.95
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - self.clipping_val, max_value=1 + self.clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss
        return loss

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_ppo(episode):

    loss = []
    agent = PPO(3, 5)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 5))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 5))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
        loss.append(score)
    return loss


if __name__ == '__main__':

    ep = 100
    loss = train_ppo(ep)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
