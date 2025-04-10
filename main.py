import gymnasium as gym
import neat
import numpy as np
#import visualize  # Необязательно, можно убрать
import os
import pickle

# Оценка одного генома (одного агента)
def eval_genome(genome, config):
    env = gym.make("BipedalWalker-v3")
    observation, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_reward = 0.0
    done = False
    while not done:
        action = net.activate(observation)  # Предсказание
        action = np.clip(action, -1, 1)  # Действия должны быть в диапазоне [-1, 1]
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    return total_reward

# Оценка всей популяции
def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

# Основной запуск
def run():
    config_path = "config-feedforward.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Создание популяции
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))  # Вывод логов
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Запуск NEAT
    winner = population.run(eval_population, n=50)

    # Сохраняем лучшего
    with open("winner_genomev2.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\n✅ Лучший агент сохранён как 'winner_genome.pkl'")

    return winner, config

if __name__ == "__main__":
    winner, config = run()
