import gymnasium as gym
import neat
import numpy as np
import pickle
from neat import ParallelEvaluator
import multiprocessing
import csv
from draw_net import draw_net

# Оценка генома (одного агента)
def eval_genome(genome, config):
    env = gym.make("BipedalWalker-v3")
    observation, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_reward = 0.0
    done = False
    steps = 0
    idle_steps = 0
    previous_x = env.unwrapped.hull.position.x

    while not done and steps < 5000:
        action = net.activate(observation)
        action = np.clip(action, -1, 1)
        
        observation, reward, terminated, truncated, _ = env.step(action)

        # увеличение награды за движение вперёд
        current_x = env.unwrapped.hull.position.x
        reward += (current_x - previous_x) * 5.0
        previous_x = current_x
        
        # Проверка на бездействие
        if np.linalg.norm(action) < 0.01:
            idle_steps += 1
            if idle_steps > 50:
                break
        else:
            idle_steps = 0

        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()
    return total_reward

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
    pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(pe.evaluate, n=3)

    # Сохраняем лучшего
    with open("winner_genomev.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nЛучший агент сохранён как 'winner_genome.pkl'")

    mean_fitness = stats.get_fitness_mean()
    best_fitness = [g.fitness for g in stats.most_fit_genomes]

    with open("log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness"])
        for i, (best, avg) in enumerate(zip(best_fitness, mean_fitness)):
            writer.writerow([i, best, avg])

    return winner, config

if __name__ == "__main__":
    winner, config = run()
    draw_net(config, winner, view=False, filename="winner_net", node_names=None)
