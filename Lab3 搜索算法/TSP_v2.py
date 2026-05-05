import math
import random
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt

# 设置后端与字体
matplotlib.use('Agg')
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def read_tsp_file(file_path):
    coords = {}
    dimension = 0
    node_coord_section = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
            elif line.startswith('EOF'):
                break
            elif node_coord_section:
                parts = line.split()
                city_id = int(parts[0]) - 1
                coords[city_id] = (float(parts[1]), float(parts[2]))
    return coords, dimension


def calculate_distance_matrix(coords, dimension):
    dist_matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i == j: continue
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_matrix[i][j] = math.sqrt(dx ** 2 + dy ** 2)
    return dist_matrix


class GeneticAlgorithmTSP:
    def __init__(self, dist_matrix, coords, pop_size=100, max_iter=1000,
                 crossover_rate=0.8, mutation_rate=0.3, elite_size=5):
        self.dist_matrix = dist_matrix
        self.coords = coords
        self.num_cities = len(dist_matrix)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.best_individual = None
        self.best_distance = float('inf')
        self.fitness_history = []

    def _calculate_distance(self, individual):
        dist = 0
        for i in range(self.num_cities):
            dist += self.dist_matrix[individual[i]][individual[(i + 1) % self.num_cities]]
        return round(dist)

    # --- 关键改进：2-opt 局部搜索 ---
    def _two_opt(self, route, max_attempts=500):
        best_route = route.copy()
        improved = True
        attempts = 0

        while improved and attempts < max_attempts:
            improved = False
            for i in range(1, self.num_cities - 2):
                for j in range(i + 1, self.num_cities):
                    if j - i == 1: continue

                    # 计算交换前后的距离变化
                    # A -> B, C -> D 变为 A -> C, B -> D
                    A, B = best_route[i - 1], best_route[i]
                    C, D = best_route[j - 1], best_route[j] if j < self.num_cities else best_route[0]

                    old_dist = self.dist_matrix[A][B] + self.dist_matrix[C][D]
                    new_dist = self.dist_matrix[A][C] + self.dist_matrix[B][D]

                    if new_dist < old_dist:
                        best_route[i:j] = reversed(best_route[i:j])
                        improved = True
                        attempts += 1
            if not improved: break
        return best_route

    def _create_individual(self):
        # 混合初始化：1个贪心解 + 其余随机
        # (可选，但为了演示GA能力，这里仍保持随机，但在solve中加入2-opt)
        ind = list(range(self.num_cities))
        random.shuffle(ind)
        return ind

    def _tournament_selection(self, population, fitnesses):
        indices = random.sample(range(len(population)), 3)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        return population[best_idx]

    def _order_crossover(self, parent1, parent2):
        size = self.num_cities
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]

        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
        return child

    def _mutation(self, individual):
        # 保持逆转变异
        if random.random() < 0.5:
            a, b = sorted(random.sample(range(self.num_cities), 2))
            individual[a:b] = reversed(individual[a:b])
        else:
            # 交换变异
            a, b = random.sample(range(self.num_cities), 2)
            individual[a], individual[b] = individual[b], individual[a]
        return individual

    def solve(self):
        population = [self._create_individual() for _ in range(self.pop_size)]

        for iter_idx in range(self.max_iter):
            # 1. 计算适应度
            distances = [self._calculate_distance(ind) for ind in population]
            fitnesses = [1.0 / d for d in distances]

            # 2. 更新全局最优
            min_dist = min(distances)
            if min_dist < self.best_distance:
                self.best_distance = min_dist
                self.best_individual = population[distances.index(min_dist)].copy()

                # --- 关键：仅对当前找到的最优解进行深度 2-opt 优化 ---
                self.best_individual = self._two_opt(self.best_individual, max_attempts=1000)
                self.best_distance = self._calculate_distance(self.best_individual)

            self.fitness_history.append(self.best_distance)

            # 3. 进化
            new_population = []
            # 保留精英
            pop_with_fitness = list(zip(population, fitnesses))
            pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
            for i in range(self.elite_size):
                new_population.append(pop_with_fitness[i][0].copy())

            # 生成后代
            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                if random.random() < self.crossover_rate:
                    child = self._order_crossover(p1, p2)
                else:
                    child = p1.copy()

                if random.random() < self.mutation_rate:
                    child = self._mutation(child)

                # 概率性对新个体进行轻量级 2-opt，增加种群质量
                if random.random() < 0.05:
                    child = self._two_opt(child, max_attempts=20)

                new_population.append(child)

            population = new_population

            if (iter_idx + 1) % 50 == 0:
                print(f"迭代 {iter_idx + 1}/{self.max_iter} | 当前最优路径长度: {self.best_distance:}")

        print(f"\n求解完成！最终最优路径长度: {self.best_distance}")
        return self.best_individual, self.best_distance

    def plot_result(self):
        plt.figure(figsize=(12, 9))
        x = [self.coords[i][0] for i in range(self.num_cities)]
        y = [self.coords[i][1] for i in range(self.num_cities)]
        plt.scatter(x, y, c='black', s=20, zorder=5)

        best_path = self.best_individual
        for i in range(self.num_cities):
            from_node = best_path[i]
            to_node = best_path[(i + 1) % self.num_cities]
            plt.plot([self.coords[from_node][0], self.coords[to_node][0]],
                     [self.coords[from_node][1], self.coords[to_node][1]],
                     'r-', linewidth=1, alpha=0.8)

        plt.title(f"TSP最优路径 (长度: {self.best_distance})")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, linestyle='--', alpha=0.7)
        filename = f"qa194_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    start_time = time.perf_counter()
    tsp_file_path = "qa194.tsp"
    POP_SIZE = 200
    MAX_ITER = 2000
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.3
    ELITE_SIZE = 10

    print(f"正在读取文件: {tsp_file_path}")
    coords, dimension = read_tsp_file(tsp_file_path)
    print(f"城市数量: {dimension}")

    print("正在计算距离矩阵...")
    dist_matrix = calculate_distance_matrix(coords, dimension)

    print("\n开始运行遗传算法...")
    print(f"种群数量：{POP_SIZE} 迭代次数：{MAX_ITER} 交叉概率：{CROSSOVER_RATE} 变异概率：{MUTATION_RATE} 精英保留数：{ELITE_SIZE}")
    ga = GeneticAlgorithmTSP(
        dist_matrix=dist_matrix,
        coords=coords,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        elite_size=ELITE_SIZE
    )
    best_path, best_distance = ga.solve()
    end_time = time.perf_counter()

    # print("\n最优路径（城市编号）:")
    # print(" → ".join([str(city + 1) for city in best_path]) + " → " + str(best_path[0] + 1))
    print(end_time - start_time," s")
    ga.plot_result()