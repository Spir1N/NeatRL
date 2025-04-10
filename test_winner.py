import pickle
import neat
import gymnasium as gym
import numpy as np
import imageio

# Загрузка winner'а
with open("winner_genomev.pkl", "rb") as f:
    winner = pickle.load(f)

# Загрузка конфигурации NEAT
config_path = "config-feedforward.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Создание среды с отрисовкой
env = gym.make("BipedalWalker-v3", render_mode="rgb_array") 

# Создание нейросети из генома победителя
net = neat.nn.FeedForwardNetwork.create(winner, config)

frames = []

obs, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    action = net.activate(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    frame = env.render()
    frames.append(frame)
    done = terminated or truncated
    total_reward += reward

imageio.mimsave("agent_run.gif", frames, fps=30)

print(f"🎯 Итоговая награда агента: {total_reward:.2f}")
env.close()
