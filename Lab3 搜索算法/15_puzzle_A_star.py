import heapq
import time

TARGET = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)

TARGET_POS = {num : (i // 4,i % 4) for i,num in enumerate(TARGET)}

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
            cur_x,cur_y = idx / 4,idx % 4
            target_x,target_y = TARGET_POS[num]
            h += abs(target_x - cur_x) + abs(target_y - cur_y)
        return h
    
    def __lt__(self,other):
        return self.f < other.f

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

def A_star(test):
    # 首先判断是否有解
    if not is_solvable(test):
        print("该测试样例无解")
        return None
    
    start_node = Node(test)
    open_heap = []
    heapq.heappush(open_heap,start_node)
    closed_set = set()

    while open_heap:
        cur_node = heapq.heappop(open_heap)
        if(cur_node.state == TARGET):
            print_solution(cur_node)
            return cur_node
        if cur_node.state in closed_set:
            continue
        closed_set.add(cur_node.state)

        for neighbor in get_neighbors(cur_node):
            if neighbor.state not in closed_set:
                heapq.heappush(open_heap, neighbor)

    print("未找到解")
    return None

if __name__ == '__main__':

    test = (2,13,4,10,6,14,0,5,3,1,8,12,9,15,7,11)
    print("初始状态")
    for i in range(4):
        row = test[i * 4 : (i + 1) * 4]
        print([x for x in row])
    print()

    start_time = time.perf_counter()
    A_star(test)
    end_time = time.perf_counter()
    print("{:.6f} s consumed".format(end_time - start_time))