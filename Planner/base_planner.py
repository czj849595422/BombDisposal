import os
import json
import random
from datetime import datetime

class BasePlanner:
    def __init__(self, env, data_dir):
        self.env = env
        self.data_dir = data_dir
        self.num_agents = env.num_players
        
        self.agent_actions = {}
        for i in range(self.num_agents):
            self.agent_actions[f'agent_{i}'] = {
                'last_action': None,
                'last_action_time': None,
                'last_action_completed': True,
                'current_action': None,
                'current_action_time': None,
                'current_action_in_progress': False,
                'action_queue': []
            }
        
        self.idle_agents = []
        self.action_history = []
        self.current_time = 0
        
        # Token使用统计
        self.token_stats = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'agent_tokens': {},
            'memory_tokens': {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }
    
    def _check_agent_idle(self, agent_id):
        agent_info = self.agent_actions[agent_id]
        env_agent_info = self.env.players[agent_id]
        
        if env_agent_info['status'] == 'room' and not agent_info['current_action_in_progress'] and not agent_info['action_queue']:
            if agent_id not in self.idle_agents:
                self.idle_agents.append(agent_id)
            return True
        return False
    
    def _generate_random_action(self, agent_id):
        agent_info = self.env.players[agent_id]
        current_room = agent_info['position']
        
        if current_room in self.env.bombs_status and self.env.bombs_status[current_room] > 0:
            return 5
        
        possible_actions = [0]
        
        room_info = self.env.adjacency[current_room]
        for i, neighbor in enumerate(room_info.keys()):
            if i < 4:
                possible_actions.append(i + 1)
        
        return random.choice(possible_actions)
    
    def _assign_action(self, agent_id):
        action = self._generate_random_action(agent_id)
        
        self.agent_actions[agent_id]['action_queue'].append(action)
        
        if agent_id in self.idle_agents:
            self.idle_agents.remove(agent_id)
    
    def get_actions(self, obs, step):
        self.current_time = step
        
        # 步骤1: 检查所有智能体是否已经完成动作并修改状态
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and agent_data['current_action_in_progress']:
                # 检查智能体是否在拆弹
                is_defusing = any(agent == agent_id for room, agent in self.env.defusing_agents.items())
                
                # 如果智能体不在拆弹，或者拆弹已经完成
                if not is_defusing and env_agent_info.get('defusing_time', 0) == 0:
                    agent_data['last_action'] = agent_data['current_action']
                    agent_data['last_action_time'] = agent_data['current_action_time']
                    agent_data['last_action_completed'] = True
                    agent_data['current_action'] = None
                    agent_data['current_action_time'] = None
                    agent_data['current_action_in_progress'] = False
        
        # 步骤2: 添加空闲智能体到空闲列表
        self.idle_agents = []
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and not agent_data['current_action_in_progress'] and not agent_data['action_queue'] and env_agent_info.get('defusing_time', 0) == 0:
                self.idle_agents.append(agent_id)
        
        # 步骤3: 为空闲智能体动作队列添加动作
        for agent_id in self.idle_agents:
            self._assign_action(agent_id)
        
        # 步骤4: 取出动作并修改状态
        actions = self._get_actions_and_update_status()
        
        self._save_action_history(actions)
        
        return actions
    
    def _update_action_status(self, actions):
        pass
    
    def _save_action_history(self, actions):
        history_entry = {
            'time': self.current_time,
            'actions': actions
        }
        self.action_history.append(history_entry)
    
    def _get_actions_and_update_status(self):
        """取出动作并修改状态"""
        actions = {}
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and agent_data['action_queue']:
                actions[agent_id] = agent_data['action_queue'].pop(0)
                agent_data['current_action'] = actions[agent_id]
                agent_data['current_action_time'] = self.current_time
                agent_data['current_action_in_progress'] = True
            else:
                actions[agent_id] = 0
        return actions
    
    def _process_valid_plan(self, actions):
        """处理有效计划
        
        Args:
            actions: 智能体动作字典
        """
        # 将高级动作转换为低级动作
        for agent_id, action_str in actions.items():
            low_level_actions = self._convert_high_level_to_low_level(agent_id, action_str)
            self.agent_actions[agent_id]['action_queue'].extend(low_level_actions)
            self.agent_actions[agent_id]['high_level_actions'].append(action_str)
            # 更新current_high_level字段
            self.agent_actions[agent_id]['current_high_level'] = action_str
        
        self.current_plan = actions
        # 规划成功，清除上一轮规划失败的反馈信息
        if hasattr(self, 'last_plan_feedback'):
            self.last_plan_feedback = None
    
    def get_action_info(self):
        info = {}
        for agent_id, agent_data in self.agent_actions.items():
            info[agent_id] = {
                'last_action': agent_data['last_action'],
                'last_action_time': agent_data['last_action_time'],
                'last_action_completed': agent_data['last_action_completed'],
                'current_action': agent_data['current_action'],
                'current_action_time': agent_data['current_action_time'],
                'current_action_in_progress': agent_data['current_action_in_progress'],
                'action_queue_length': len(agent_data['action_queue']),
                'is_idle': agent_id in self.idle_agents
            }
        return info
    
    def save_action(self, step):
        step_dir = os.path.join(self.data_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)
        
        action_info = self.get_action_info()
        
        with open(os.path.join(step_dir, 'action_info.json'), 'w', encoding='utf-8') as f:
            json.dump(action_info, f, indent=2, ensure_ascii=False)
    
    def update_token_usage(self, input_tokens, output_tokens, agent_id=None):
        """更新token使用情况"""
        self.token_stats['input_tokens'] += input_tokens
        self.token_stats['output_tokens'] += output_tokens
        self.token_stats['total_tokens'] += input_tokens + output_tokens
        
        if agent_id:
            if agent_id not in self.token_stats['agent_tokens']:
                self.token_stats['agent_tokens'][agent_id] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0
                }
            self.token_stats['agent_tokens'][agent_id]['input_tokens'] += input_tokens
            self.token_stats['agent_tokens'][agent_id]['output_tokens'] += output_tokens
            self.token_stats['agent_tokens'][agent_id]['total_tokens'] += input_tokens + output_tokens
    
    def update_memory_token_usage(self, input_tokens, output_tokens):
        """更新记忆token使用情况"""
        self.token_stats['memory_tokens']['input_tokens'] += input_tokens
        self.token_stats['memory_tokens']['output_tokens'] += output_tokens
        self.token_stats['memory_tokens']['total_tokens'] += input_tokens + output_tokens
    
    def get_token_stats(self):
        """获取token使用情况统计"""
        # 计算agent平均使用量
        agent_count = len(self.token_stats['agent_tokens'])
        if agent_count > 0:
            total_agent_tokens = sum(agent['total_tokens'] for agent in self.token_stats['agent_tokens'].values())
            average_agent_tokens = total_agent_tokens / agent_count
        else:
            average_agent_tokens = 0
        
        return {
            'input_tokens': self.token_stats['input_tokens'],
            'output_tokens': self.token_stats['output_tokens'],
            'total_tokens': self.token_stats['total_tokens'],
            'average_agent_tokens': average_agent_tokens,
            'agent_tokens': self.token_stats['agent_tokens'],
            'memory_tokens': self.token_stats['memory_tokens']
        }
    
    def get_token_stats_text(self):
        """获取token使用情况的文本描述"""
        stats = self.get_token_stats()
        text = []
        text.append(f"Token使用情况统计：")
        text.append(f"- 输入token总量：{stats['input_tokens']}")
        text.append(f"- 输出token总量：{stats['output_tokens']}")
        text.append(f"- 总token使用量：{stats['total_tokens']}")
        text.append(f"- Agent平均使用量：{stats['average_agent_tokens']:.2f}")
        
        # 添加记忆token统计
        memory_tokens = stats['memory_tokens']
        if memory_tokens['total_tokens'] > 0:
            text.append("\n记忆Token使用情况：")
            text.append(f"- 输入token：{memory_tokens['input_tokens']}")
            text.append(f"- 输出token：{memory_tokens['output_tokens']}")
            text.append(f"- 总token：{memory_tokens['total_tokens']}")
        
        if stats['agent_tokens']:
            text.append("\n各Agent使用情况：")
            for agent_id, agent_stats in stats['agent_tokens'].items():
                text.append(f"- {agent_id}：")
                text.append(f"  输入token：{agent_stats['input_tokens']}")
                text.append(f"  输出token：{agent_stats['output_tokens']}")
                text.append(f"  总token：{agent_stats['total_tokens']}")
        
        return '\n'.join(text)
    
    def _convert_high_level_to_low_level(self, agent_id, action_str):
        """将高级动作转换为低级动作列表"""
        import re
        agent_info = self.env.players[agent_id]
        current_room = agent_info['position']
        
        if action_str.startswith('WAIT'):
            match = re.match(r'WAIT (\d+)', action_str)
            if match:
                wait_time = int(match.group(1))
                return [0] * wait_time
            return [0]
        elif action_str == 'Disposal':
            return [5]
        elif action_str.startswith('MOVE'):
            match = re.match(r'MOVE (\d+)', action_str)
            if match:
                target_room = int(match.group(1))
                room_info = self.env.adjacency[current_room]
                for i, neighbor in enumerate(room_info.keys()):
                    if neighbor == target_room:
                        return [i + 1]
        return [0]
    
    def _get_idle_agents(self):
        """获取当前空闲的智能体列表"""
        idle_agents = []
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and not agent_data['current_action_in_progress'] and not agent_data['action_queue'] and env_agent_info.get('defusing_time', 0) == 0:
                idle_agents.append(agent_id)
        return idle_agents
    
    def get_agents_status_text(self):
        """获取所有智能体当前动作状态的文本描述"""
        status_text = []
        for agent_id, agent_info in self.env.players.items():
            # 获取智能体名称
            agent_name = agent_id
            if hasattr(self, 'agent_names') and agent_id in self.agent_names:
                agent_name = self.agent_names[agent_id]
            
            current_room = agent_info['position']
            status = agent_info['status']
            
            # 检查智能体是否在通道中移动
            if status == 'corridor':
                target_room = agent_info.get('target_room')
                remaining_time = agent_info.get('remaining_time', 0)
                status_str = f"{agent_name}: 从房间{current_room}前往房间{target_room}（还需{remaining_time}分钟）"
            else:
                # 检查智能体是否在拆弹
                is_defusing = any(agent == agent_id for room, agent in self.env.defusing_agents.items())
                if is_defusing:
                    for room, agent in self.env.defusing_agents.items():
                        if agent == agent_id:
                            defusing_room = room
                            break
                    remaining_time = agent_info.get('defusing_time', 0)
                    status_str = f"{agent_name}: 在房间{defusing_room}拆弹中（还需{remaining_time}分钟）"
                else:
                    # 检查智能体的动作队列和当前动作
                    agent_data = self.agent_actions.get(agent_id, {})
                    current_action = agent_data.get('current_action')
                    action_queue = agent_data.get('action_queue', [])
                    
                    # 检查是否有当前高级动作
                    current_high_level = agent_data.get('current_high_level')
                    recent_action = current_high_level
                    
                    # 如果没有当前高级动作，尝试从high_level_actions获取最近的动作
                    if not recent_action:
                        high_level_actions = agent_data.get('high_level_actions', [])
                        if high_level_actions:
                            # 检查high_level_actions的最后一个元素是字典还是字符串
                            last_action = high_level_actions[-1]
                            if isinstance(last_action, dict) and 'action' in last_action:
                                recent_action = last_action['action']
                            else:
                                recent_action = last_action
                    
                    if recent_action:
                        if isinstance(recent_action, str):
                            if recent_action.startswith('MOVE'):
                                target_room = recent_action.split()[1]
                                # 计算移动所需时间
                                move_time = self.env.adjacency[current_room].get(int(target_room), 1)
                                status_str = f"{agent_name}: 在房间{current_room}，准备前往房间{target_room}（需要{move_time}分钟）"
                            elif recent_action == 'Disposal':
                                status_str = f"{agent_name}: 在房间{current_room}，准备拆弹"
                            elif recent_action.startswith('WAIT'):
                                wait_time = recent_action.split()[1]
                                status_str = f"{agent_name}: 在房间{current_room}，等待{wait_time}分钟"
                            else:
                                status_str = f"{agent_name}: 在房间{current_room}，准备执行{recent_action}"
                        else:
                            status_str = f"{agent_name}: 在房间{current_room}，准备执行动作"

                    elif current_action is not None:
                        if current_action == 0:
                            status_str = f"{agent_name}: 在房间{current_room}，静止"
                        elif current_action == 5:
                            status_str = f"{agent_name}: 在房间{current_room}，拆弹中"
                        elif 1 <= current_action <= 4:
                            # 获取移动目标房间
                            room_info = self.env.adjacency[current_room]
                            neighbors = list(room_info.keys())
                            if current_action <= len(neighbors):
                                target_room = neighbors[current_action - 1]
                                move_time = room_info[target_room]
                                status_str = f"{agent_name}: 从房间{current_room}前往房间{target_room}（还需{move_time - 1}分钟）"
                            else:
                                status_str = f"{agent_name}: 在房间{current_room}，静止"
                        else:
                            status_str = f"{agent_name}: 在房间{current_room}，执行动作{current_action}"
                    elif action_queue:
                        next_action = action_queue[0]
                        if next_action == 0:
                            status_str = f"{agent_name}: 在房间{current_room}，准备静止"
                        elif next_action == 5:
                            status_str = f"{agent_name}: 在房间{current_room}，准备拆弹"
                        elif 1 <= next_action <= 4:
                            # 获取移动目标房间
                            room_info = self.env.adjacency[current_room]
                            neighbors = list(room_info.keys())
                            if next_action <= len(neighbors):
                                target_room = neighbors[next_action - 1]
                                move_time = room_info[target_room]
                                status_str = f"{agent_name}: 在房间{current_room}，准备前往房间{target_room}（需要{move_time}分钟）"
                            else:
                                status_str = f"{agent_name}: 在房间{current_room}，准备静止"
                        else:
                            status_str = f"{agent_name}: 在房间{current_room}，准备执行动作{next_action}"
                    else:
                        # 空闲状态
                        status_str = f"{agent_name}: 在房间{current_room}，空闲"
            
            status_text.append(status_str)
        
        return '\n'.join(status_text)