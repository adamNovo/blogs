import gym
import pandas as pd
import tensorflow as tf

def get_best_action(state):
    state_df = pd.DataFrame(columns=["cart_pos", "cart_vel", "pole_ang", "pole_vel"])
    state_df.loc[len(state_df),:] = state
    q = model.predict(state_df.values, batch_size=1)
    action = pd.Series(q[0]).idxmax()
    return action, q

env = gym.make("CartPole-v1")
no_actions = env.action_space.n
no_observations = env.observation_space.shape[0]
print(no_actions)
print(no_observations)

#########
## Random
#########

observation = env.reset()
for ep in range(10):
    state = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

#########
## Trained
#########

# model = tf.keras.models.load_model("model.h5")

# for ep in range(10):
#     state = env.reset()
#     r_total = 0
#     for t in range(1000):
#         env.render()
#         action, q = get_best_action(state)
#         next_state, reward, done, info = env.step(action)
#         r_total += reward
#         if done:
#             print(f"Episode {ep} finished after {t} timesteps")
#             break
#         state = next_state
# env.close()