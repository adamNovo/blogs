import numpy as np
import pandas as pd
import random
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam

class DeepAgent(object):
    def __init__(self, env, epsilon=0.9, epsilon_min=0.1, epsilon_decay=0.995,
                 experiences_size=2000):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.experiences = deque(maxlen=experiences_size)
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        print("Action space {0}".format(self.action_size))
        print("Observation space {0}".format(self.state_size))

    def build_nn(self, layer_1, layer_2, learning_rate):
        model = Sequential()
        model.add(Dense(layer_1, input_dim=self.state_size, activation="relu"))
        if layer_2 > 0:
            model.add(Dense(layer_2, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        self.model = model

    def add_experience(self, state, action, reward, next_state, done):
        self.experiences.append((state, action, reward, next_state, done))

    def get_action(self, state, force_exploit=False):
        if force_exploit or np.random.rand() > self.epsilon:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])  # returns action
        else:
            return random.randrange(self.action_size)

    def learn_vec(self, batch_size, gamma=0.95, epochs=10, min_experience=1000,
                  verbose=False):
        if min_experience > len(self.experiences):
            return
        minibatch = random.sample(self.experiences, batch_size)
        minibatch = np.vstack(minibatch)
        states = np.vstack(minibatch[:,0])
        actions = np.vstack(minibatch[:,1])
        targets = minibatch[:,2]
        next_states = np.vstack(minibatch[:,3])
        dones = minibatch[:,4]
        predicts_next = self.model.predict(next_states)
        targets[np.where(np.logical_not(dones))] = (targets + gamma *
            np.amax(predicts_next, axis=1))[np.where(np.logical_not(dones))]
        target_fs = self.model.predict(states)
        i = np.arange(target_fs.shape[0])
        target_fs[i, actions.flatten()] = targets
        hist = self.model.fit(states, target_fs, batch_size=batch_size,
                              epochs=epochs, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if verbose:
            print("Total loss: {}. Epsilon: {}".format(
                sum(hist.history["loss"]), self.epsilon))


    def save_model(self, filename):
        self.model.save(filename + ".h5")

    def load_model(self, filename):
        self.model = load_model(filename + ".h5")

    def save_weights(self, filename):
        target = open(filename + ".txt", 'w')
        for layer in self.model.layers:
            weights = layer.get_weights() # list of numpy arrays
            target.write(str(weights))
            target.write("\n")
        target.close()
