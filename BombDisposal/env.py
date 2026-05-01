import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import random

class BombDisposalEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, task_file=None, num_players=2, time_limit=100, seed=None, bomb_defuse_time=5):
        self.num_players = num_players
        self.time_limit = time_limit
        self.task_file = task_file
        self.seed = seed
        self.bomb_defuse_time = bomb_defuse_time
        self.defusing_agents = {}
        
        if seed:
            random.seed(seed)
        
        if task_file:
            self.load_task(task_file)
        else:
            self.generate_task()
        
        self.reset()
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            'global': spaces.Dict({
                'bombs': spaces.Dict({}),
                'rooms': spaces.Dict({}),
                'adjacency': spaces.Dict({}),
                'players': spaces.Dict({})
            }),
            'local': spaces.Dict({})
        })
    
    def generate_task(self):
        A = random.randint(10, 19)
        M = random.randint(10, 20)
        N = random.randint(10, 20)
        
        room_positions = {}
        for i in range(1, A + 1):
            while True:
                x = random.randint(0, M - 1)
                y = random.randint(0, N - 1)
                if (x, y) not in room_positions.values():
                    room_positions[i] = (x, y)
                    break
        
        adjacency = {i: {} for i in range(1, A + 1)}
        
        for room in range(1, A + 1):
            current_connections = len(adjacency[room])
            needed_connections = random.randint(2, 4) - current_connections
            
            if needed_connections > 0:
                possible_neighbors = [r for r in range(1, A + 1) 
                                   if r != room 
                                   and r not in adjacency[room]
                                   and len(adjacency[r]) < 4]
                
                if possible_neighbors:
                    num_to_add = min(needed_connections, len(possible_neighbors))
                    neighbors = random.sample(possible_neighbors, num_to_add)
                    
                    for neighbor in neighbors:
                        distance = random.randint(2, 9)
                        adjacency[room][neighbor] = distance
                        adjacency[neighbor][room] = distance
        
        for room in range(1, A + 1):
            if len(adjacency[room]) < 2:
                possible_neighbors = [r for r in range(1, A + 1) 
                                   if r != room 
                                   and r not in adjacency[room]
                                   and len(adjacency[r]) < 4]
                
                if possible_neighbors:
                    neighbor = random.choice(possible_neighbors)
                    distance = random.randint(2, 9)
                    adjacency[room][neighbor] = distance
                    adjacency[neighbor][room] = distance
        
        B = random.randint(2, min(5, A - 1))
        bomb_rooms = random.sample([r for r in range(1, A + 1)], B)
        bombs = {room: 1 for room in bomb_rooms}
        
        start_room = random.choice([r for r in range(1, A + 1) if r not in bomb_rooms])
        
        self.M = M
        self.N = N
        self.A = A
        self.B = B
        self.room_positions = room_positions
        self.adjacency = adjacency
        self.bombs = bombs
        self.start_room = start_room
    
    def load_task(self, task_file):
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        self.M = task['M']
        self.N = task['N']
        self.A = task['A']
        self.B = task['B']
        self.room_positions = {int(k): v for k, v in task['room_positions'].items()}
        self.adjacency = {int(k): {int(n): d for n, d in v.items()} 
                         for k, v in task['adjacency'].items()}
        self.bombs = {int(k): v for k, v in task['bombs'].items()}
        self.start_room = task['start_room']
    
    def save_task(self, file_path):
        task = {
            'M': self.M,
            'N': self.N,
            'A': self.A,
            'B': self.B,
            'room_positions': self.room_positions,
            'adjacency': self.adjacency,
            'bombs': self.bombs,
            'start_room': self.start_room
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(task, f, indent=2)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.time_step = 0
        self.explored_rooms = set()
        self.explored_rooms.add(self.start_room)
        self.defusing_agents = {}
        
        self.bombs_status = {room: self.bomb_defuse_time for room in self.bombs.keys()}
        
        self.players = {}
        for i in range(self.num_players):
            if self.start_room in self.bombs_status:
                observation = self.bombs_status[self.start_room]
            else:
                observation = -1
            self.players[f'agent_{i}'] = {
                'position': self.start_room,
                'status': 'room',
                'target_room': None,
                'distance_to_target': 0,
                'observation': observation,
                'defusing_time': 0
            }
        
        return self.get_obs(), {}
    
    def get_obs(self, mode='global'):
        if mode == 'global':
            rooms_info = {}
            for room in range(1, self.A + 1):
                room_players = [agent for agent, info in self.players.items() if info['position'] == room]
                if room in self.bombs_status:
                    bomb_status = self.bombs_status[room]
                else:
                    bomb_status = -1
                rooms_info[str(room)] = {
                    'players': room_players,
                    'bomb_status': bomb_status,
                    'explored': room in self.explored_rooms,
                    'neighbors': list(self.adjacency[room].keys())
                }
            
            players_info = {}
            for agent, info in self.players.items():
                players_info[agent] = {
                    'position': info['position'],
                    'status': info['status'],
                    'target_room': info['target_room'],
                    'distance_to_target': info['distance_to_target'],
                    'observation': info['observation']
                }
            
            bombs_info = {}
            for room, status in self.bombs_status.items():
                bombs_info[str(room)] = status
            
            return {
                'bombs': bombs_info,
                'rooms': rooms_info,
                'adjacency': {str(room): {str(neighbor): distance for neighbor, distance in neighbors.items()}
                             for room, neighbors in self.adjacency.items()},
                'players': players_info
            }
        else:
            local_obs = {}
            for agent, info in self.players.items():
                room = info['position']
                if room in self.bombs_status:
                    bomb_status = self.bombs_status[room]
                else:
                    bomb_status = -1
                room_info = {
                    'room_id': room,
                    'bomb_status': bomb_status,
                    'explored': room in self.explored_rooms,
                    'neighbors': list(self.adjacency[room].keys())
                }
                local_obs[agent] = {
                    'player_info': info,
                    'room_info': room_info
                }
            return local_obs
    
    def step(self, actions):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        self.time_step += 1
        
        for agent, info in self.players.items():
            if info['status'] == 'corridor':
                info['distance_to_target'] -= 1
                if info['distance_to_target'] <= 0:
                    info['position'] = info['target_room']
                    info['status'] = 'room'
                    info['target_room'] = None
                    info['distance_to_target'] = 0
                    # 更新观察状态为当前房间的炸弹状态
                    room = info['position']
                    if room in self.bombs_status:
                        info['observation'] = self.bombs_status[room]
                    else:
                        info['observation'] = -1
                    continue
            
            if info['defusing_time'] > 0:
                continue
            
            action = actions.get(agent, 0)
            
            if action == 0:
                continue
            elif 1 <= action <= 4:
                room = info['position']
                neighbors = list(self.adjacency[room].keys())
                if len(neighbors) >= action:
                    target_room = neighbors[action - 1]
                    distance = self.adjacency[room][target_room]
                    info['status'] = 'corridor'
                    info['target_room'] = target_room
                    info['distance_to_target'] = distance
            elif action == 5:
                room = info['position']
                if room in self.bombs_status and self.bombs_status[room] > 0:
                    if room in self.defusing_agents:
                        action = 0
                    else:
                        self.defusing_agents[room] = agent
                        info['defusing_time'] = self.bomb_defuse_time
        
        for agent, info in self.players.items():
            room = info['position']
            if room not in self.explored_rooms:
                self.explored_rooms.add(room)
                reward += 10
        
        for agent, info in self.players.items():
            if info['defusing_time'] > 0:
                info['defusing_time'] -= 1
                room = info['position']
                if room in self.bombs_status:
                    self.bombs_status[room] -= 1
                    info['observation'] = self.bombs_status[room]
                if info['defusing_time'] == 0:
                    room = info['position']
                    if room in self.bombs_status:
                        reward += 100
                    if room in self.defusing_agents:
                        del self.defusing_agents[room]
        
        if all(status == 0 for status in self.bombs_status.values()):
            terminated = True
            reward += 500
        
        if self.time_step >= self.time_limit:
            truncated = True
        
        return self.get_obs(), reward, terminated, truncated, info
    
    def get_direction(self, agent_id, target_room):
        agent_info = self.players.get(f'agent_{agent_id}')
        if not agent_info:
            return -1
        
        current_room = agent_info['position']
        if target_room not in self.adjacency[current_room]:
            return -1
        
        neighbors = list(self.adjacency[current_room].keys())
        for i, neighbor in enumerate(neighbors):
            if neighbor == target_room:
                return i + 1
        
        return -1
    
    def print_adjacency_matrix(self):
        print("邻接矩阵（房间 - 邻居:距离）:")
        for room in sorted(self.adjacency.keys()):
            neighbors = self.adjacency[room]
            neighbor_str = ", ".join([f"{n}:{d}" for n, d in sorted(neighbors.items())])
            print(f"  房间 {room}: [{neighbor_str}]")
    
    def task_generate(self, save_path=None):
        self.generate_task()
        if save_path:
            self.save_task(save_path)
        return {
            'M': self.M,
            'N': self.N,
            'A': self.A,
            'B': self.B,
            'room_positions': self.room_positions,
            'adjacency': self.adjacency,
            'bombs': self.bombs,
            'start_room': self.start_room
        }