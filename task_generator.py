import json
import random
import os

# 确保task目录存在
task_dir = './task'
os.makedirs(task_dir, exist_ok=True)

# 生成符合要求的任务
def generate_task(seed):
    random.seed(seed)
    
    # 房间数量10-15个
    A = random.randint(10, 15)
    
    # 炸弹数量3-5个
    B = random.randint(3, 5)
    
    # 生成房间位置
    room_positions = {}
    M = 20
    N = 20
    
    for i in range(1, A + 1):
        while True:
            x = random.randint(0, M - 1)
            y = random.randint(0, N - 1)
            if (x, y) not in room_positions.values():
                room_positions[i] = (x, y)
                break
    
    # 生成邻接矩阵，确保连通性且度数不超过3
    adjacency = {i: {} for i in range(1, A + 1)}
    degrees = {i: 0 for i in range(1, A + 1)}
    
    # 生成最小生成树确保连通性
    visited = set()
    queue = [1]  # 从房间1开始
    visited.add(1)
    
    while len(visited) < A:
        current = queue.pop(0)
        
        # 找到未访问的房间
        unvisited = [room for room in range(1, A + 1) if room not in visited]
        
        # 选择一个未访问的房间连接
        if unvisited:
            target = random.choice(unvisited)
            distance = random.randint(3, 10)
            
            # 检查度数
            if degrees[current] < 3 and degrees[target] < 3:
                adjacency[current][target] = distance
                adjacency[target][current] = distance
                degrees[current] += 1
                degrees[target] += 1
                visited.add(target)
                queue.append(target)
    
    # 随机添加一些额外的边，增加连通性
    while True:
        room1 = random.randint(1, A)
        room2 = random.randint(1, A)
        
        if room1 != room2 and room2 not in adjacency[room1] and degrees[room1] < 3 and degrees[room2] < 3:
            distance = random.randint(3, 10)
            adjacency[room1][room2] = distance
            adjacency[room2][room1] = distance
            degrees[room1] += 1
            degrees[room2] += 1
        
        # 检查是否所有房间的度数都达到3或者无法添加更多边
        if all(degrees[room] >= 3 for room in degrees):
            break
        
        # 限制循环次数
        if random.random() > 0.3:
            break
    
    # 生成炸弹状态（初始状态为需要拆除的时间）
    bombs = {}
    bomb_rooms = random.sample(range(1, A + 1), B)
    for room in bomb_rooms:
        # 炸弹初始状态为需要拆除的时间（5分钟）
        bombs[room] = 5
    
    # 选择一个无炸弹的房间作为起始房间
    non_bomb_rooms = [room for room in range(1, A + 1) if room not in bombs]
    start_room = random.choice(non_bomb_rooms)
    
    # 验证连通性
    def is_connected():
        visited = set()
        queue = [1]
        visited.add(1)
        
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == A
    
    # 验证房间度数
    def check_degrees():
        for room in adjacency:
            if len(adjacency[room]) > 3:
                return False
        return True
    
    # 验证距离范围
    def check_distances():
        for room in adjacency:
            for neighbor, distance in adjacency[room].items():
                if distance < 3 or distance > 10:
                    return False
        return True
    
    # 验证所有条件
    if not is_connected() or not check_degrees() or not check_distances():
        return None
    
    # 构建任务数据
    task = {
        'A': A,  # 房间数量
        'B': B,  # 炸弹数量
        'M': M,  # 建筑宽度
        'N': N,  # 建筑长度
        'room_positions': room_positions,
        'adjacency': adjacency,
        'bombs': bombs,
        'start_room': start_room
    }
    
    return task

# 生成30个符合条件的任务
def generate_tasks():
    tasks_generated = 0
    seeds_used = []
    
    while tasks_generated < 30:
        seed = random.randint(100000, 999999)
        
        if seed not in seeds_used:
            task = generate_task(seed)
            
            if task:
                # 保存任务文件
                filename = os.path.join(task_dir, f'{seed}.json')
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(task, f, indent=2, ensure_ascii=False)
                
                print(f'生成任务 {tasks_generated + 1}/30: {seed}.json')
                tasks_generated += 1
                seeds_used.append(seed)

if __name__ == '__main__':
    generate_tasks()
    print('任务生成完成！')