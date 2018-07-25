import gym
import numpy as np
import pandas as pd
import time
from collections import deque
from gym import wrappers

from DeepAgent import DeepAgent
from EnvSetup import EnvSetup

class RunEnv(object):
    def __init__(self, env_name, verbose=False):
        self.env_name = env_name
        self.verbose = verbose
        self.current_exp = deque(maxlen=1001)

    def train(self, episodes, upload_to_gym=False, show_video=False):
        env_params = EnvSetup()
        env_params = env_params.get_params(self.env_name)
        self.build_env(folder=self.env_name, show_video=show_video)
        self.build_agent()
        if self.env_name == "LunarLander-v2":
            self.run_lunar(episodes=episodes, gamma=env_params["gamma"],
                           epochs=env_params["epochs"],
                           min_experience=env_params["min_experience"],
                           upload_to_gym=upload_to_gym)
        elif self.env_name == "CartPole-v0":
            self.run_cart_pole(episodes=episodes, gamma=env_params["gamma"],
                               epochs=env_params["epochs"],
                               min_experience=env_params["min_experience"],
                               upload_to_gym=upload_to_gym)

    def build_env(self, folder, show_video):
        self.env = gym.make(self.env_name)
        self.env = wrappers.Monitor(self.env, folder,
            video_callable=lambda count: count % 100 == 0 and show_video,
            force=True)
        self.state_size = self.env.observation_space.shape[0]

    def build_agent(self, train=True):
        env_params = EnvSetup()
        env_params = env_params.get_params(self.env_name)
        self.agent = DeepAgent(self.env, epsilon=env_params["epsilon"],
                               epsilon_min=env_params["epsilon_min"],
                               epsilon_decay=env_params["epsilon_decay"],
                               experiences_size=env_params["experiences_size"])
        if train:
            self.agent.build_nn(layer_1=env_params["layer_1"],
                                layer_2=env_params["layer_2"],
                                learning_rate=env_params["learning_rate"])


    def run_lunar(self, episodes, gamma, epochs, min_experience,
                  upload_to_gym=False):
        total_land = 0
        for i in range(episodes):
            self.current_exp = deque(maxlen=1001)
            # init state
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # run
            time_t = 0
            exploit = False
            if i > episodes - 200:
                # exploit last 10% of episodes
                exploit = True
            while True:
                time_t += 1
                action = self.agent.get_action(state, force_exploit=exploit)
                new_state, reward, done, info = self.env.step(action)
                new_state = np.reshape(new_state, [1, self.state_size])
                self.agent.add_experience(state, action, reward, new_state, done)
                self.current_exp.append((state, action, reward, new_state, done))
                if reward == 100:
                    total_land += 1
                """
                repeat_good_experience = False
                repeat_good_run = False
                # repeat successful last experience
                if reward == 100 and repeat_good_experience:
                    repeat = 10
                    print("Repeating good experience {} times".format(repeat))
                    for i in range(repeat):
                        self.agent.add_experience(state, action, reward, new_state, done)
                # repeat successful run in memory
                if reward == 100 and repeat_good_run:
                    repeat = 15
                    print("Repeating good full run {} times".format(repeat))
                    for i in range(repeat):
                        for state, action, reward, new_state, done in self.current_exp:
                            self.agent.add_experience(state, action, reward, new_state, done)
                """
                state = new_state
                average_score = np.mean(self.env.get_episode_rewards()[-100:])
                if done:
                    print("""\n{}/{}. Exploit: {}. Time: {}.
                             Landed: {}.
                             C reward: {}. Last 100: {}.
                             Final reward: {}""".format(
                             i, episodes, exploit, time_t, total_land,
                             round(self.env.get_episode_rewards()[-1], 2),
                             round(np.mean(self.env.get_episode_rewards()[-100:]), 2),
                             reward))
                    break
            if not exploit and i % 1 == 0:
                #print("Updating model {}".format(i))
                self.agent.learn_vec(batch_size=32, gamma=gamma,
                                     epochs=epochs,
                                     min_experience=min_experience,
                                     verbose=self.verbose)
            if (i + 1) % 100 == 0:
                print("Saving model")
                self.agent.save_model(filename=self.env_name + "/model")
        # save final model
        self.agent.save_model(filename=self.env_name + "/model")
        self.agent.save_weights(self.env_name + "/model")
        self.env.close()
        if upload_to_gym:
            gym.upload(self.env_name, api_key="sk_9Gt38t5ATla5HL7QI8rFTA")

    def run_cart_pole(self, episodes, gamma, epochs, min_experience,
                      upload_to_gym=False):
        for i in range(episodes):
            # init state
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # run
            time_t = 0
            exploit = False
            if i > episodes - 200:
                # exploit last 10% of episodes
                exploit = True
            while True:
                time_t += 1
                action = self.agent.get_action(state, force_exploit=exploit)
                #action = self.env.env.action_space.sample()
                new_state, reward, done, info = self.env.step(action)
                new_state = np.reshape(new_state, [1, self.state_size])
                reward = reward if not done else -10
                self.agent.add_experience(state, action, reward, new_state, done)
                state = new_state
                if done:
                    print("""\n{}/{}. Exploit: {}. Time: {}.
                             Rewards: {}.
                             Final reward: {}""".format(
                             i, episodes, exploit, time_t,
                             self.env.get_episode_rewards()[-1], reward))
                    break
            if not exploit:
                self.agent.learn_vec(batch_size=32, gamma=gamma,
                                     epochs=epochs,
                                     min_experience=min_experience,
                                     verbose=self.verbose)
            if (i + 1) % 100 == 0:
                print("Saving model")
                self.agent.save_model(filename=self.env_name + "/model")
        # save final model
        self.agent.save_model(filename=self.env_name + "/model")
        self.agent.save_weights(self.env_name + "/model")
        self.env.close()
        if upload_to_gym:
            gym.upload(self.env_name, api_key="sk_9Gt38t5ATla5HL7QI8rFTA")

    def test(self, episodes, upload_to_gym=False, show_video=True):
        print("Evaluating model")
        self.build_env(folder=self.env_name + "_test", show_video=show_video)
        self.build_agent(train=False)
        self.agent.load_model(self.env_name + "/model")
        self.agent.save_weights(self.env_name + "_test" + "/model")
        # run test using leaded recent model
        for i in range(episodes):
            # init state
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # run
            time_t = 0
            while True:
                time_t += 1
                action = self.agent.get_action(state, exploit=True)
                new_state, reward, done, info = self.env.step(action)
                new_state = np.reshape(new_state, [1, self.state_size])
                state = new_state
                if done:
                    print("Test episode: {}/{}. Time {}".format(
                        i, episodes, time_t))
                    break
        self.env.close()
        if upload_to_gym:
            gym.upload(self.env_name + "_test", api_key="sk_9Gt38t5ATla5HL7QI8rFTA")


if __name__ == "__main__":
    env_names = ["LunarLander-v2", "CartPole-v0"]
    select_env = 0 # 0=Lunar, 1=CartPole

    env_params = EnvSetup()
    env_params = env_params.get_params(env_name=env_names[select_env])
    print("Running {}".format(env_names[select_env]))
    print("Params: {}".format(env_params))
    start = time.time()
    run_env = RunEnv(env_name=env_names[select_env], verbose=True)
    run_env.train(episodes=env_params["episodes"],
                  upload_to_gym=True, show_video=False)
    #run_env.test(episodes=env_params["episodes"],
    #             upload_to_gym=True, show_video=False)
    end = time.time()
    print("Total time: {}".format(end - start))
    print("Finished with params: {}".format(env_params))
