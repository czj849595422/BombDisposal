import os
import json
import re
import random
from datetime import datetime
from .sync_planner import SyncPlanner
from openai import OpenAI
from utils import to_dict

class OraclePlanner(SyncPlanner):
    def __init__(self, env, data_dir=None, debug=False, model_name=None):
        super().__init__(env, data_dir)
        self.debug = debug
        self.agent_names = self._assign_agent_names()
        self.action_history = []
        self.model_history = []
        self.feedback = None
        self.call_count = 0
        
        # 增加高级动作指令记录和当前高级动作
        for agent_id in self.agent_actions:
            self.agent_actions[agent_id]['high_level_actions'] = []
            self.agent_actions[agent_id]['current_high_level'] = None
        
        # 加载模型配置
        self.model_config = self._load_model_config()
        self.model_name = model_name if model_name else self.model_config.get('default_model', 'gpt-3.5-turbo')
    
    def get_actions(self, obs, step):
        self.current_time = step
        
        # 步骤1: 检测并更新智能体的动作状态
        self._update_agent_action_status()
        
        # 步骤2: 添加空闲智能体到空闲列表
        self.idle_agents = []
        for agent_id in self.agent_actions:
            agent_data = self.agent_actions[agent_id]
            env_agent_info = self.env.players[agent_id]
            
            if env_agent_info['status'] == 'room' and not agent_data['current_action_in_progress'] and not agent_data['action_queue']:
                self.idle_agents.append(agent_id)
        
        # 步骤3: 检查是否所有智能体都空闲
        if len(self.idle_agents) == len(self.agent_actions):
            self.all_idle = True
        else:
            self.all_idle = False
        
        # 步骤4: 只有当所有智能体都空闲时，才为所有智能体安排动作
        if self.all_idle:
            if self.debug:
                # Debug模式：生成模拟的大模型指令并按照真实流程处理
                self._generate_debug_instructions()
            else:
                # 调用大模型生成指令
                self._call_model_for_instructions(step, obs)
        
        # 步骤5: 取出动作并修改状态
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
        
        self._save_action_history(actions)
        
        return actions
    
    def _generate_debug_instructions(self):
        """Debug模式：生成模拟的大模型指令并按照真实流程处理"""
        # 获取当前时间和系统提示
        step = self.current_time
        system_prompt = self.get_system_prompt(self.env.get_obs(mode='global'))
        user_prompt = self.get_user_prompt()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建调用记录
        call_record = {
            'step': step,
            'timestamp': timestamp,
            'call_count': self.call_count,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
        
        # 生成模拟的大模型指令
        high_level_actions = {}
        for agent_id in self.idle_agents:
            agent_info = self.env.players[agent_id]
            current_room = agent_info['position']
            
            # 根据策略生成高级动作指令
            if current_room in self.env.bombs_status and self.env.bombs_status[current_room] > 0:
                action_str = "Disposal"
            else:
                room_info = self.env.adjacency[current_room]
                if room_info:
                    neighbors = list(room_info.keys())
                    neighbor = random.choice(neighbors)
                    action_str = f"MOVE {neighbor}"
                else:
                    action_str = "WAIT 1"
            
            high_level_actions[agent_id] = action_str
        
        # 将高级动作转换为模拟的大模型指令
        instruction_lines = ['EXECUTE']
        for agent_id, action_str in high_level_actions.items():
            name = self.agent_names[agent_id]
            instruction_lines.append(f"NAME {name} ACTION {action_str}")
        instruction = '\n'.join(instruction_lines)
        
        # 解析指令（按照真实流程）
        parsed_actions = self._parse_action_instruction(instruction)
        
        # 验证动作合法性（按照真实流程）
        valid_actions = {}
        errors = []
        
        for agent_id, action_str in parsed_actions.items():
            is_valid, error_msg = self._validate_action(agent_id, action_str)
            if is_valid:
                low_level_actions = self._convert_high_level_to_low_level(agent_id, action_str)
                valid_actions[agent_id] = low_level_actions
                # 记录高级动作指令
                self.agent_actions[agent_id]['high_level_actions'].append({
                    'step': self.current_time,
                    'action': action_str,
                    'time': self.current_time
                })
            else:
                errors.append(f"{self.agent_names[agent_id]}: {error_msg}")
        
        if errors:
            # 规划失败，生成反馈
            self.feedback = "规划失败：" + "；".join(errors)
            # 所有智能体静止
            for agent_id in self.idle_agents:
                self.agent_actions[agent_id]['action_queue'].append(0)
        else:
            # 规划成功，清除反馈
            self.feedback = None
            # 设置动作队列
            for agent_id, actions in valid_actions.items():
                self.agent_actions[agent_id]['action_queue'].extend(actions)
        
        # 保存调用记录
        call_record['response'] = instruction
        call_record['feedback'] = self.feedback
        call_record['instruction'] = instruction
        call_record['usage'] = None
        
        # 添加到历史记录
        history=f"[Time {step}] Oracle: {instruction}"
        self.model_history.append(history)
        
        # 保存到文件
        if self.data_dir:
            step_dir = os.path.join(self.data_dir, f'step_{step}')
            os.makedirs(step_dir, exist_ok=True)
            record_file = os.path.join(step_dir, f'oracle_{step}_{timestamp}_{self.call_count}.json')
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(call_record, f, indent=2, ensure_ascii=False)
        
        self.idle_agents = []
    
    def _call_model_for_instructions(self, step, obs):
        """调用大模型生成指令"""
        system_prompt = self.get_system_prompt(obs)
        user_prompt = self.get_user_prompt()
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")
        
        # 保存调用记录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        call_record = {
            'step': step,
            'timestamp': timestamp,
            'call_count': self.call_count,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
        
        if self.debug:
            # Debug模式：使用模拟回答
            response = self._convert_action_to_instruction({})
            usage = None
            for agent_id in self.idle_agents:
                agent_info = self.env.players[agent_id]
                current_room = agent_info['position']
                
                if current_room in self.env.bombs_status and self.env.bombs_status[current_room] > 0:
                    action = 5
                else:
                    room_info = self.env.adjacency[current_room]
                    if room_info:
                        neighbors = list(room_info.keys())
                        neighbor = random.choice(neighbors)
                        for i, n in enumerate(room_info.keys()):
                            if n == neighbor:
                                action = i + 1
                                break
                    else:
                        action = 0
                
                self.agent_actions[agent_id]['action_queue'].append(action)
        else:
            # 实际调用大模型
            response, usage = self._call_language_model(system_prompt, user_prompt, max_retries=3)
            
            # 解析指令
            parse_success, high_level_actions, parse_feedback = self._parse_action_instruction(response)
            
            # 验证动作合法性
            valid_actions = {}
            errors = []
            
            # 添加解析错误到错误列表
            if not parse_success and parse_feedback:
                errors.append(parse_feedback)
            
            for agent_id, action_str in high_level_actions.items():
                is_valid, error_msg = self._validate_action(agent_id, action_str)
                if is_valid:
                    low_level_actions = self._convert_high_level_to_low_level(agent_id, action_str)
                    valid_actions[agent_id] = low_level_actions
                    # 记录高级动作指令
                    self.agent_actions[agent_id]['high_level_actions'].append({
                        'step': step,
                        'action': action_str,
                        'time': self.current_time
                    })
                    # 更新current_high_level字段
                    self.agent_actions[agent_id]['current_high_level'] = action_str
                else:
                    errors.append(f"{self.agent_names[agent_id]}: {error_msg}")
            
            if errors:
                # 规划失败，生成反馈
                self.feedback = "规划失败：" + "；".join(errors)
                self.call_count += 1
                # 所有智能体静止
                for agent_id in self.idle_agents:
                    self.agent_actions[agent_id]['action_queue'].append(0)
            else:
                # 规划成功，清除反馈
                self.feedback = None
                self.call_count=0
                # 设置动作队列
                for agent_id, actions in valid_actions.items():
                    self.agent_actions[agent_id]['action_queue'].extend(actions)
        
        # 保存调用记录
        call_record['response'] = response if 'response' in locals() else None
        call_record['feedback'] = self.feedback
        call_record['instruction'] = response if 'response' in locals() else None
        call_record['usage'] = usage if 'usage' in locals() else None
        
        # 添加到历史记录
        history=f"[Time {step}] Oracle: {response}"
        self.model_history.append(history)
        
        if self.data_dir:
            step_dir = os.path.join(self.data_dir, f'step_{step}')
            os.makedirs(step_dir, exist_ok=True)
            record_file = os.path.join(step_dir, f'oracle_{step}_{timestamp}_{self.call_count}.json')
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(call_record, f, indent=2, ensure_ascii=False)
        
        self.idle_agents = []
    
    def _call_language_model(self, system_prompt, user_prompt, agent_id=None, max_retries=3):
        """调用大模型，支持最大尝试调用次数"""
        # 获取模型配置

        print(f"system_prompt: {system_prompt}")
        print(f"user_prompt: {user_prompt}")

        model_info = self.model_config['models'].get(self.model_name)
        instruction=None
        for attempt in range(max_retries):
            try:
                # 获取API密钥
                api_key = model_info.get('api_key')
                
                # 创建OpenAI客户端
                client_kwargs = {
                    'api_key': api_key
                }
                if model_info.get('api_base'):
                    client_kwargs['base_url'] = model_info['api_base']
                
                client = OpenAI(**client_kwargs)
                
                # 调用大模型
                response = client.chat.completions.create(
                    model=model_info.get('model', 'gpt-3.5-turbo'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=model_info.get('temperature', 0.1),
                    max_tokens=model_info.get('max_tokens', 512)
                )
                
                # 提取回答内容
                instruction = response.choices[0].message.content.strip()
                print(f"response: {instruction}")
                
                # 获取token使用统计
                usage=to_dict(response.usage)
                print(usage)
                
                # 更新token使用情况
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    self.update_token_usage(input_tokens, output_tokens, agent_id)

                # 检查是否包含EXECUTE关键字
                if 'EXECUTE' in instruction:
                    return instruction, usage
                else:
                    # 如果回答不包含EXECUTE，继续重试
                    print(f"第{attempt + 1}次调用失败：回答格式不正确")
                    
            except Exception as e:
                print(f"第{attempt + 1}次调用失败：{str(e)}")

        return instruction, None
    
    def get_actions_info(self):
        """获取所有角色动作信息的文字表达"""
        info = {}
        for agent_id, data in self.agent_actions.items():
            name = self.agent_names[agent_id]
            
            info[name] = {
                'position': self.env.players[agent_id]['position'],
                'status': self.env.players[agent_id]['status'],
                'last_action': data['last_action'],
                'last_action_time': data['last_action_time'],
                'last_action_completed': data['last_action_completed'],
                'current_action': data['current_action'],
                'current_action_time': data['current_action_time'],
                'current_action_in_progress': data['current_action_in_progress'],
                'high_level_actions': data['high_level_actions']
            }
        return info
    
    def get_actions_instruction(self):
        """获取指令输出形式、指令动作、示例"""
        return {
            'output_format': 'EXECUTE\nNAME {name} ACTION {action}\nNAME {name} ACTION {action}\n...',
            'valid_actions': [
                'WAIT {time}',
                'MOVE {room}',
                'Disposal'
            ],
            'example': 'EXECUTE\nNAME Alice ACTION MOVE 5\nNAME Bob ACTION Disposal\nNAME Charlie ACTION WAIT 2'
        }
    
    def get_obs_str(self, obs):
        """获取文字描述的obs，从obs中提取信息"""
        obs_str = []
        
        # 地图信息
        obs_str.append(f"地图信息：共{len(obs.get('rooms', {}))}个房间")
        
        # 房间联通情况与距离
        room_connections = []
        adjacency = obs.get('adjacency', {})
        for room in sorted(adjacency.keys()):
            connections = []
            for neighbor, distance in adjacency[room].items():
                connections.append(f"房间{neighbor}({distance}分钟）")
            connections_str = ", ".join(connections) if connections else "无"
            room_connections.append(f"房间{room}：{connections_str}")
        obs_str.append(f"房间联通情况：\n{';\n'.join(room_connections)}")
        
        # 时间信息
        obs_str.append(f"当前时间：{self.current_time}/{self.env.time_limit}")
        
        # 炸弹拆除进度
        bombs = obs.get('bombs', {})
        total_bombs = len(bombs)
        removed_bombs = sum(1 for status in bombs.values() if status == 0)
        obs_str.append(f"炸弹拆除进度：{removed_bombs}/{total_bombs}")
        
        # 角色位置或状态
        agents_info = []
        players = obs.get('players', {})
        for agent_id, info in players.items():
            name = self.agent_names[agent_id]
            if info['status'] == 'room':
                observation = info.get('observation', -1)
                if observation == -1:
                    obs_desc = "无炸弹"
                elif observation == 0:
                    obs_desc = "炸弹已拆除"
                elif observation == self.env.bomb_defuse_time:
                    obs_desc = "炸弹未拆除"
                elif observation > 0:
                    obs_desc = "炸弹正在拆除"
                else:
                    obs_desc = f"观察值：{observation}"
                agents_info.append(f"{name}在房间{info['position']}（{obs_desc}）")
            else:
                agents_info.append(f"{name}正在前往房间{info['target_room']}（剩余{info['distance_to_target']}分钟）")
        obs_str.append(f"角色状态：{'；'.join(agents_info)}")
        
        return '\n'.join(obs_str)
    
    def get_history(self, k=None):
        """获取大模型历史记录，可支持仅输出最近k条历史"""
        if k is None:
            return self.model_history
        return self.model_history[-k:]
    
    def get_feedback(self):
        """获取历史反馈信息"""
        return self.feedback
    
    def get_rules(self):
        """获取游戏规则"""
        rules = """
拆弹小队需要在一栋建筑内部进行拆弹工作，需要所有人通过协作完成。
建筑中存在若干房间，房间之间存在通道，玩家可以消耗若干分钟通过通道移动到相邻房间。
你需要指挥玩家进行行动，尽快检查房间并拆除炸弹。
"""
        return rules
    
    def get_actions_instruction_text(self):
        """获取动作执行信息"""
        instructions="""
对于每个智能体，你可以执行以下动作：
1. WAIT {time}：等待time分钟（1-10）
2. MOVE {room}：移动到编号为{room}的相邻房间
3. Disposal：拆除当前房间的炸弹（需要5分钟）

每次发出指令，你需要先输出'EXECUTE'，然后输出 NAME {agent} ACTION {action}。
例如：‘
EXECUTE
NAME Alice ACTION MOVE 5
NAME Bob ACTION Disposal
NAME Charlie ACTION WAIT 2’
"""
        return instructions
    
    def get_agents_info_text(self):
        """获取角色动作信息"""
        agents_info = []
        agents_info.append("\n角色动作信息：")
        actions_info = self.get_actions_info()
        for name, info in actions_info.items():
            if info['status'] == 'room':
                agents_info.append(f"- {name}：在房间{info['position']}")
            else:
                agents_info.append(f"- {name}：正在前往房间{info['target_room']}（需要{info['distance_to_target']}分钟）")
        return '\n'.join(agents_info)
    
    def get_history_text(self, k=None):
        """获取历史记录文本"""
        history = self.get_history(k)
        history = '\n'.join(history)
        if history:
            return f"\n历史记录：\n{history}"
        return ""
    
    def get_feedback_text(self):
        """获取反馈信息文本"""
        feedback = self.get_feedback()
        if feedback:
            return f"\n反馈信息：{feedback}"
        return ""
    
    def get_system_prompt(self, obs):
        """获取系统提示，从obs中提取信息"""
        system_prompt = []
        
        # 添加各个部分
        system_prompt.append(self.get_rules())
        system_prompt.append(self.get_actions_instruction_text())
        system_prompt.append(self.get_agents_info_text())
        
        # 添加智能体状态信息
        system_prompt.append("\n智能体状态：")
        system_prompt.append(self.get_agents_status_text())
        
        # 环境观察（从obs中提取）
        system_prompt.append("\n环境观察：")
        system_prompt.append(self.get_obs_str(obs))
        
        system_prompt.append(self.get_history_text())
        system_prompt.append(self.get_feedback_text())
        
        return '\n'.join(system_prompt)
    
    def get_user_prompt(self):
        """获取用户提示"""
        user="""请一步步思考，给出简洁的分析，并为所有智能体下达指令：
"""
        return user