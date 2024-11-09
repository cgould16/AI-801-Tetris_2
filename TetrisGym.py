import os
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import random
from collections import deque


def create_dqn(input_shape, num_actions):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Register the environment
os.environ["ALE_ROM_PATH"] = "/roms/Tetris.nes"
gym.register_envs(ale_py)
env_train = gym.make('ALE/Tetris-v5')
obs_shape = env_train.observation_space.shape
num_actions = env_train.action_space.n
dqn = create_dqn(input_shape=obs_shape, num_actions=num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.MeanSquaredError()

# Experience replay buffer
replay_buffer = deque(maxlen=10000)

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env_train.action_space.sample()  # Random action
    else:
        # Process state and predict Q-values
        state = np.expand_dims(state / 255.0, axis=0)  # Normalize pixel values and add batch dimension
        q_values = dqn(state)
        return np.argmax(q_values[0])  # Best action

def sample_from_replay(batch_size):
    samples = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*samples)
    return (
        np.array(states) / 255.0,  # Normalize
        np.array(actions),
        np.array(rewards, dtype=np.float32),
        np.array(next_states) / 255.0,  # Normalize
        np.array(dones, dtype=np.float32)
    )


num_episodes = 500
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial epsilon for epsilon-greedy policy
epsilon_decay = 0.995
min_epsilon = 0.1

for episode in range(num_episodes):
    state, _ = env_train.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env_train.step(action)
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # Only train if the replay buffer has enough samples
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_from_replay(batch_size)

            # Calculate Q-values and targets
            next_q_values = tf.reduce_max(dqn(next_states), axis=1)
            targets = rewards + gamma * next_q_values * (1 - dones)

            with tf.GradientTape() as tape:
                q_values = tf.reduce_sum(dqn(states) * tf.one_hot(actions, num_actions), axis=1)
                loss = loss_fn(targets, q_values)

            grads = tape.gradient(loss, dqn.trainable_variables)
            optimizer.apply_gradients(zip(grads, dqn.trainable_variables))

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env_test = gym.make('ALE/Tetris-v5', render_mode="human")
state = env_test.reset()[0]
done = False
while not done:
    state = np.expand_dims(state / 255.0, axis=0)  # Normalize state
    action = np.argmax(dqn.predict(state)[0])
    next_state, reward, done, _, _ = env_test.step(action)
    state = next_state
    env_test.render()

env_test.close()