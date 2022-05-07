#!/usr/bin/python

import gym 
import numpy as np
import tensorflow.compat.v1 as tf

tf.reset_default_graph()
tf.disable_eager_execution()


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

def discount_rewards(rewards, discount_rate):
    discount_rewv = np.empty(len(rewards))
    cum_reward = 0
    for ri in reversed(range(len(rewards))):
        cum_reward = rewards[ri] + cum_reward * discount_rate
        discount_rewv[ri] = cum_reward
    return discount_rewv

logits = tf.keras.layers.Dense(2, activation='softmax')


X = tf.placeholder(dtype=tf.float32, shape=(None, 4))
hidden = tf.keras.layers.Dense(4, activation='elu')(X)
logits = tf.keras.layers.Dense(2, activation='softmax')(hidden)
action = tf.argmax(logits, axis=1)
y = 1. - tf.to_float(logits)

xentroy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer()
grads_and_vars = optimizer.compute_gradients(xentroy)
gradients = [grad for grad, _ in grads_and_vars]
gradient_phs = []
grads_and_vars_feed = []
for grad, var in grads_and_vars:
    grad_ph = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_phs.append(grad_ph)
    grads_and_vars_feed.append((grad, var))

train_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()

n_iteration = 200
n_game_per_update = 10
n_max_step = 1000

def discount_normalizer_rewards(rewards):
    


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    with tf.Session() as sess:
        sess.run(init)
        obs = env.reset()
        for g in range(n_iteration):
            all_rewards = []
            all_grads = []
            for ep in range(n_game_per_update):
                ep_rewards = 0
                obs = env.reset()
                cur_rewards = 0
                cur_grads = []
                for _ in range(n_max_step):
                    act = sess.run(action, feed_dict={X: np.reshape(obs, (-1, 4))})
                    grad = sess.run(gradients,  feed_dict={X: np.reshape(obs, (-1, 4))})
                    cur_grads.append(grad)
                    env.render()
                    obs, reward, done, info = env.step(act[0])
                    cur_rewards += reward
                    if done:
                        break
                all_rewards.append(cur_rewards)
                all_grads.append(cur_grads)
            #  update gradient

        env.close()

