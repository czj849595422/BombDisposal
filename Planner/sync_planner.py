import random
import os
import json
import re

from .base_planner import BasePlanner


class SyncPlanner(BasePlanner):
    def __init__(self, env, data_dir=None):
        super().__init__(env, data_dir)
        self.all_idle = False
    
    def _assign_agent_names(self):
        names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
        agent_names = {}
        for i, agent_id in enumerate(self.agent_actions.keys()):
            agent_names[agent_id] = names[i % len(names)]
        return agent_names
    
    def _load_model_config(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到模型配置文件 {config_path}，使用默认配置")
            return {
                'models': {
                    'deepseek': {
                        'provider': 'deepseek',
                        'model': 'deepseek-chat',
                        'api_base': 'https://api.deepseek.com',
                        'api_key_env': 'sk-6f90a0fc704347b2af39daf9eaeb51c3',
                        'temperature': 0.7,
                        'max_tokens': 500
                    }
                },
                'default_model': 'deepseek'
            }
    
    def _parse_action_instruction(self, instruction):
        """解析大模型指令为具体动作，返回(解析结果, 解析动作, 反馈信息)"""
        actions = {}
        feedback = ""
        success = True
        
        if not instruction or 'EXECUTE' not in instruction:
            feedback = "指令格式错误：缺少EXECUTE标记"
            success = False
            return success, actions, feedback
        
        lines = instruction.split('\n')
        # 记录每个房间的拆弹指令
        disposal_rooms = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('NAME '):
                match = re.match(r'NAME (\w+) ACTION (.+)', line)
                if match:
                    name = match.group(1)
                    action_str = match.group(2)
                    agent_id = self._get_agent_id_by_name(name)
                    if agent_id:
                        # 检查是否为同一智能体重复安排动作
                        if agent_id in actions:
                            error_msg = f"禁止多个智能体同时拆除同一颗炸弹。"
                            print(f"警告: {error_msg}")
                            feedback += error_msg + "；"
                            success = False
                            continue
                        # 检查是否为非空闲智能体安排动作
                        if hasattr(self, 'idle_agents') and agent_id not in self.idle_agents:
                            error_msg = f"智能体{name}不是空闲状态，忽略动作安排"
                            print(f"警告: {error_msg}")
                            feedback += error_msg + "；"
                            success = False
                            continue
                        # 检查是否有多个智能体在同一房间拆弹
                        if action_str == 'Disposal':
                            agent_info = self.env.players[agent_id]
                            current_room = agent_info['position']
                            
                            # 检查该房间的炸弹是否已经有人正在拆除
                            if current_room in self.env.defusing_agents:
                                existing_agent = self.env.defusing_agents[current_room]
                                error_msg = f"房间{current_room}的炸弹已由智能体{self.agent_names.get(existing_agent, existing_agent)}正在拆除，禁止{name}同时拆除该炸弹。"
                                print(f"警告: {error_msg}")
                                feedback += error_msg + "；"
                                success = False
                                continue
                            
                            if current_room in disposal_rooms:
                                error_msg = f"房间{current_room}已有智能体{disposal_rooms[current_room]}安排拆弹，忽略智能体{name}的拆弹指令"
                                print(f"警告: {error_msg}")
                                feedback += error_msg + "；"
                                success = False
                                continue
                            disposal_rooms[current_room] = name
                        actions[agent_id] = action_str
        
        # 检查指令数量是否与空闲智能体匹配
        if success and hasattr(self, 'idle_agents') and len(actions) != len(self.idle_agents):
            error_msg = f"EXECUTE动作指令数量({len(actions)})与空闲智能体数量({len(self.idle_agents)})不匹配"
            print(f"警告: {error_msg}")
            feedback += error_msg + "；"
            success = False
            # 为未安排动作的空闲智能体添加静止动作
            for agent_id in self.idle_agents:
                if agent_id not in actions:
                    actions[agent_id] = "WAIT 1"
        
        # 移除末尾的分号
        if feedback.endswith("；"):
            feedback = feedback[:-1]
        
        return success, actions, feedback
    
    def _convert_action_to_instruction(self, actions):
        instruction_lines = ['EXECUTE']
        
        for agent_id, action in actions.items():
            name = self.agent_names[agent_id]
            
            if action == 0:
                action_str = f"WAIT 1"
            elif action == 5:
                action_str = "Disposal"
            elif 1 <= action <= 4:
                agent_info = self.env.players[agent_id]
                current_room = agent_info['position']
                room_info = self.env.adjacency[current_room]
                neighbors = list(room_info.keys())
                if action <= len(neighbors):
                    target_room = neighbors[action - 1]
                    action_str = f"MOVE {target_room}"
                else:
                    action_str = f"WAIT 1"
            else:
                action_str = f"WAIT 1"
            
            instruction_lines.append(f"NAME {name} ACTION {action_str}")
        
        return '\n'.join(instruction_lines)
    
    def _get_agent_id_by_name(self, name):
        for agent_id, agent_name in self.agent_names.items():
            if agent_name == name:
                return agent_id
        return None
    
    def _validate_action(self, agent_id, action_str):
        agent_info = self.env.players[agent_id]
        current_room = agent_info['position']
        
        if action_str.startswith('WAIT'):
            match = re.match(r'WAIT (\d+)', action_str)
            if not match:
                return False, "WAIT动作格式错误，应为WAIT {time}"
            time = int(match.group(1))
            if time <= 0 or time > 10:
                return False, f"WAIT时间必须大于0且小于等于10，当前为{time}"
            return True, None
        
        elif action_str == 'Disposal':
            if current_room not in self.env.bombs_status:
                return False, f"房间{current_room}没有炸弹"
            if self.env.bombs_status[current_room] <= 0:
                return False, f"房间{current_room}的炸弹已拆除"
            if current_room in self.env.defusing_agents:
                return False, f"房间{current_room}已有其他智能体正在拆除炸弹"
            return True, None
        
        elif action_str.startswith('MOVE'):
            match = re.match(r'MOVE (\d+)', action_str)
            if not match:
                return False, "MOVE动作格式错误，应为MOVE {room}"
            target_room = int(match.group(1))
            if target_room not in self.env.adjacency[current_room]:
                return False, f"房间{current_room}和房间{target_room}不相邻"
            return True, None
        
        else:
            return False, f"未知动作: {action_str}"
    
    def _update_agent_action_status(self):
        """检测并更新智能体的动作状态，包括低级动作和高级动作"""
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
            
            # 更新高级动作状态
            current_high_level = agent_data.get('current_high_level')
            if current_high_level:
                if current_high_level.startswith('MOVE'):
                    parts = current_high_level.split()
                    if len(parts) == 2:
                        move_to_room = parts[1]
                        current_room = env_agent_info['position']
                        if str(current_room) == move_to_room:
                            agent_data['current_high_level'] = None
                elif current_high_level == 'Disposal':
                    if env_agent_info.get('defusing_time', 0) == 0:
                        agent_data['current_high_level'] = None
    
    def _get_idle_agents(self):
        """获取当前空闲的智能体列表"""
        idle_agents = []
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and not agent_data['current_action_in_progress'] and not agent_data['action_queue']:
                idle_agents.append(agent_id)
        return idle_agents
    
    def get_actions(self, obs, step):
        self.current_time = step
        
        # 步骤1: 检测并更新智能体的动作状态
        self._update_agent_action_status()
        
        # 步骤2: 添加空闲智能体到空闲列表
        self.idle_agents = self._get_idle_agents()
        
        # 步骤3: 检查是否所有智能体都空闲
        if len(self.idle_agents) == len(self.agent_actions):
            self.all_idle = True
        else:
            self.all_idle = False
        
        # 步骤4: 只有当所有智能体都空闲时，才为所有智能体安排动作
        if self.all_idle:
            for agent_id in self.idle_agents:
                agent_info = self.env.players[agent_id]
                current_room = agent_info['position']
                
                # 如果当前房间有炸弹，优先拆弹
                if current_room in self.env.bombs_status and self.env.bombs_status[current_room] > 0:
                    self.agent_actions[agent_id]['action_queue'].append(5)
                else:
                    # 否则优先移动到相邻房间，避免一直静止
                    room_info = self.env.adjacency[current_room]
                    if room_info:
                        # 随机选择一个相邻房间移动，增加探索性
                        neighbors = list(room_info.keys())
                        import random
                        neighbor = random.choice(neighbors)
                        for i, n in enumerate(room_info.keys()):
                            if n == neighbor:
                                self.agent_actions[agent_id]['action_queue'].append(i + 1)
                                break
                    else:
                        # 如果没有相邻房间，只能静止
                        self.agent_actions[agent_id]['action_queue'].append(0)
            self.idle_agents = []
        
        # 步骤5: 取出动作并修改状态
        actions = self._get_actions_and_update_status()
        
        self._save_action_history(actions)
        
        return actions