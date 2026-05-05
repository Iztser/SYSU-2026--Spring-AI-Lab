import math
import random
import matplotlib
# 1. 先设置后端，避开PyCharm兼容性报错
matplotlib.use('Agg')
# 2. 全局设置中文字体（优先用微软雅黑，黑体兜底）
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
# 3. 解决负号显示成方块的问题
matplotlib.rcParams["axes.unicode_minus"] = False
# 4. 最后导入pyplot
import matplotlib.pyplot as plt
import numpy as np

def read_tsp_file(file_path):
    coords = {}
    dimension = 0
    node_coord_section = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
            elif line.startswith('EOF'):
                break
            elif node_coord_section:
                parts = line.split()
                city_id = int(parts[0]) - 1
                x = float(parts[1])
                y = float(parts[2])
                coords[city_id] = (x, y)

    return coords, dimension

def calculate_distance_matrix(coords, dimension):
    dist_matrix = np.zeros((dimension, dimension), dtype=int)
    for i in range(dimension):
        x1, y1 = coords[i]
        for j in range(dimension):
            if i == j:
                continue
            x2, y2 = coords[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = round(math.sqrt(dx ** 2 + dy ** 2))
            dist_matrix[i][j] = dist
    return dist_matrix

class GeneticAlgorithmTSP:
    def __init__(self, dist_matrix, coords, pop_size=100, max_iter=1000,
                 crossover_rate=0.8, mutation_rate=0.2, elite_size=2):
        self.dist_matrix = dist_matrix
        self.coords = coords
        self.num_cities = len(dist_matrix)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.best_individual = None
        self.best_fitness = -float('inf')
        self.best_distance = float('inf')
        self.fitness_history = []
    
    # 生成一个随机路径
    def _create_individual(self):
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    # 计算单个路径的总距离
    def _calculate_distance(self, individual):
        total_dist = 0
        for i in range(self.num_cities):
            from_city = individual[i]
            to_city = individual[(i + 1) % self.num_cities]
            total_dist += self.dist_matrix[from_city][to_city]
        return total_dist

    # 计算路径适应度，路径长度的倒数
    def _fitness(self, individual):
        distance = self._calculate_distance(individual)
        return 1.0 / distance if distance > 0 else 0

    # 锦标赛选择，选取三条随机路径中最好一条
    def _tournament_selection(self, population, tournament_size=3):
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: self._fitness(x), reverse=True)
        return tournament[0]

    # 顺序交叉
    def _order_crossover(self, parent1, parent2):
        size = self.num_cities
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        ptr = end
        for city in parent2:
            if city not in child:
                if ptr >= size:
                    ptr = 0
                child[ptr] = city
                ptr += 1
        return child

    # 逆转变异，随机反转一条路径
    def _inversion_mutation(self, individual):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        individual[start:end] = reversed(individual[start:end])
        return individual

    # 进化：选择-交叉-变异-保留
    def _evolve(self, population):
        # 按适应度排序
        population.sort(key=lambda x: self._fitness(x), reverse=True)
        new_population = []

        # 精英保留
        new_population.extend(population[:self.elite_size])

        # 填充剩余个体
        while len(new_population) < self.pop_size:
            # 选择父代
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # 交叉
            if random.random() < self.crossover_rate:
                child = self._order_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # 变异
            if random.random() < self.mutation_rate:
                child = self._inversion_mutation(child)

            new_population.append(child)

        return new_population

    def solve(self):
        # 初始化种群
        population = [self._create_individual() for _ in range(self.pop_size)]

        for iter in range(self.max_iter):
            # 进化
            population = self._evolve(population)

            # 更新最优解
            current_best = max(population, key=lambda x: self._fitness(x))
            current_fitness = self._fitness(current_best)
            current_distance = self._calculate_distance(current_best)

            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.best_individual = current_best.copy()
                self.best_distance = current_distance

            # 记录历史
            self.fitness_history.append(self.best_distance)

            # 打印进度
            if (iter + 1) % 50 == 0:
                print(f"迭代 {iter + 1}/{self.max_iter} | 当前最优路径长度: {self.best_distance}")

        print(f"\n求解完成！最优路径长度: {self.best_distance}")
        return self.best_individual, self.best_distance

    def plot_result(self):
        plt.figure(figsize=(10, 8))

        # 绘制城市
        x = [self.coords[i][0] for i in range(self.num_cities)]
        y = [self.coords[i][1] for i in range(self.num_cities)]
        plt.scatter(x, y, c='blue', s=50, zorder=5, edgecolors='black', linewidth=0.5)

        # 绘制路径
        best_path = self.best_individual
        for i in range(self.num_cities):
            from_city = best_path[i]
            to_city = best_path[(i + 1) % self.num_cities]
            plt.plot([self.coords[from_city][0], self.coords[to_city][0]],
                     [self.coords[from_city][1], self.coords[to_city][1]],
                     'r-', linewidth=1.5)

        plt.title(f"TSP最优路径 (长度: {self.best_distance})")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("dj38.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    tsp_file_path = "dj38.tsp"
    POP_SIZE = 150
    MAX_ITER = 1000
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    ELITE_SIZE = 2

    print(f"正在读取文件: {tsp_file_path}")
    coords, dimension = read_tsp_file(tsp_file_path)
    print(f"城市数量: {dimension}")

    print("正在计算距离矩阵...")
    dist_matrix = calculate_distance_matrix(coords, dimension)

    print("\n开始运行遗传算法...")
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


    print("\n最优路径（城市编号）:")
    print(" → ".join([str(city + 1) for city in best_path]) + " → " + str(best_path[0] + 1))


    ga.plot_result()