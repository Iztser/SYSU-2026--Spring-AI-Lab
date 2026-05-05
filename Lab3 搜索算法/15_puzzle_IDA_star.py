import time

TARGET = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)
TARGET_POS = {num : (i // 4, i % 4) for i, num in enumerate(TARGET)}
DIRECTION = [(-1,0),(1,0),(0,-1),(0,1)]

class Node:
    def __init__(self,state,parent = None,g = 0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = self.manhattan_distance()
        self.f = self.h + self.g
        self.zero_pos = self.state.index(0)

    def manhattan_distance(self):
        h = 0
        for idx,num in enumerate(self.state):
            if num == 0:
                continue
            cur_x,cur_y = idx // 4,idx % 4
            target_x,target_y = TARGET_POS[num]
            h += abs(target_x - cur_x) + abs(target_y - cur_y)
        return h

# 逆序数 与 空位的评估值 奇偶性不相同 → 可解
def is_solvable(state):
    inv = 0
    q = [x for x in state if x != 0]
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if(q[i] > q[j]):
                inv += 1
    cur_row = state.index(0) // 4
    cur_col = state.index(0) % 4
    man_dis = abs(3 - cur_row) + abs(3 - cur_col)
    return abs(inv - man_dis) % 2 

def get_neighbors(node):
    neighbors = []
    x,y = node.zero_pos // 4, node.zero_pos % 4
    state_list = list(node.state)
    for dx,dy in DIRECTION:
        new_x,new_y = x + dx,y + dy
        if 0 <= new_x < 4 and 0 <= new_y < 4:
            new_zero_pos = new_x * 4 + new_y
            old_zero_pos = node.zero_pos
            state_list[old_zero_pos],state_list[new_zero_pos] = state_list[new_zero_pos],state_list[old_zero_pos]
            new_state = tuple(state_list)
            neighbors.append(Node(new_state,node,node.g + 1))
            state_list[old_zero_pos],state_list[new_zero_pos] = state_list[new_zero_pos],state_list[old_zero_pos]
    return neighbors

def print_solution(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    print("最优解步数",len(path) - 1)
    print('-' * 20)
    for step, state in enumerate(path):
        print("第{}步".format(step))
        for i in range(4):
            row = state[i * 4: (i + 1) * 4]
            print([x for x in row])
    print('-' * 20)


def dfs(node, limit, path_set):
    # 总代价超过阈值，剪枝，返回当前f值作为候选阈值
    if node.f > limit:
        return (False, node.f, None)
    if node.state == TARGET:
        return (True, limit, node)
    
    min_next_limit = float('inf')
    neighbors = get_neighbors(node)
    neighbors.sort(key=lambda x: x.f)
    
    for neighbor in neighbors:
        if neighbor.state in path_set:
            continue
        path_set.add(neighbor.state)
        found, next_limit, sol_node = dfs(neighbor, limit, path_set)
        if found:
            return (True, next_limit, sol_node)
        if next_limit < min_next_limit:
            min_next_limit = next_limit
        path_set.remove(neighbor.state)
    
    return (False, min_next_limit, None)


def IDA_star(test):
    # 首先判断是否可解
    if not is_solvable(test):
        print("该测试样例无解")
        return None
    
    start_node = Node(test)
    limit = start_node.h
    print(f"初始搜索阈值: {limit}")
    

    while True:
        # 记录当前DFS路径上的状态,避免环
        path_set = set()
        path_set.add(start_node.state)
        found, next_limit, sol_node = dfs(start_node, limit, path_set)
        
        if found:
            print_solution(sol_node)
            return sol_node
        # 无候选阈值
        if next_limit == float('inf'):
            print("未找到解")
            return None
        # 更新阈值为本次搜索的最小超限值，继续迭代
        limit = next_limit
        print(f"更新搜索阈值为: {limit}")

if __name__ == '__main__':
    test = (2,13,4,10,6,14,0,5,3,1,8,12,9,15,7,11)
    
    print("初始状态")
    for i in range(4):
        row = test[i * 4 : (i + 1) * 4]
        print([x for x in row])
    print()
    
    start_time = time.perf_counter()
    IDA_star(test)
    end_time = time.perf_counter()
    print(f"{end_time - start_time:.6f} s consumed")