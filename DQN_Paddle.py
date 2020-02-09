from pongEnv import Pala

import random
import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import adam

# Importamos el entorno
env = Pala()
np.random.seed(0)
tensor_board = TensorBoard(log_dir='./logs')  # Para crear logs


# COMENTAR
class DQN:

    """ Implementación del algoritmo DQN """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.gamma = .95  # Factor de descuento  default 0.95
        self.epsilon = 0.5  # Nivel random del algoritmo 0 -> determinista  1 -> full exploratorio.  Default 1
        self.epsilon_min = .01  # Nivel minimo de epsilon
        self.epsilon_decay = .885  # Ratio de bajada de epsilon     default 0.995  no afecta mucho
        self.batch_size = 64
        self.learning_rate = 0.001  # Ratio de aprendizaje
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        # default, loss = mse     amsgrad -> AMSGradient variante
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate, amsgrad=True))
        return model

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


def train_dqn(episode):

    loss = []
    agent = DQN(3, 5)
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
                break
        loss.append(score)
    return loss


if __name__ == '__main__':

    n = 5
    ep = 100  # 100
    total_loss = [0]*ep
    for j in range(n):
        # ep = ep
        loss = train_dqn(ep)
        total_loss = [x + y for x, y in zip(total_loss, loss)]
        # plt.plot([i for i in range(ep)], loss)
        # plt.xlabel('episodes')
        # plt.ylabel('reward')
        # plt.show()
    mean_loss = [x / n for x in total_loss]
    plt.plot([i for i in range(ep)], mean_loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
