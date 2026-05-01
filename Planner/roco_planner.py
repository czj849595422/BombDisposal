import os
import json
import re
import random
from datetime import datetime
from .oracle_planner import OraclePlanner
from openai import OpenAI
from utils import to_dict


class RoCoPlanner(OraclePlanner):
    def __init__(self, env, data_dir=None, debug=False, model_name=None, max_communication_per_minute=6):
        super().__init__(env, data_dir, debug, model_name)
        self.max_communication_per_minute = max_communication_per_minute
        self.current_communication_count = 0  # 当前交流次数
        self.agent_speech_count = {}  # 每个智能体本轮发言次数
        self.current_speaker_index = 0  # 当前发言者索引
        self.round_history = []  # 本轮规划的发言记录
        self.history_speeches = []  # 历史发言记录
        self.current_plan = None  # 当前计划
        self.last_plan_feedback = None  # 上一轮规划失败的反馈信息
        
        # 增加高级动作指令记录
        for agent_id in self.agent_actions:
            self.agent_actions[agent_id]['high_level_actions'] = []
            self.agent_speech_count[agent_id] = 0
    
    def _assign_agent_names(self):
        names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
        agent_names = {}
        for i, agent_id in enumerate(self.agent_actions.keys()):
            agent_names[agent_id] = names[i % len(names)]
        return agent_names
    
    def _convert_action_to_instruction(self, actions):
        """将具体动作转换为指令格式"""
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
        """根据名字获取智能体ID"""
        for agent_id, agent_name in self.agent_names.items():
            if agent_name == name:
                return agent_id
        return None
    
    def _validate_action(self, agent_id, action_str):
        """验证动作合法性"""
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
        
        # 步骤4: 只有当所有智能体都空闲时，才进行RoCo规划
        if self.all_idle:
            if self.debug:
                # Debug模式：生成模拟的RoCo规划
                self._generate_debug_roco_plan()
            else:
                # 执行RoCo规划流程
                self._execute_roco_planning(step, obs)
        
        # 步骤5: 取出动作并修改状态
        actions = self._get_actions_and_update_status()
        
        self._save_action_history(actions)
        
        return actions
    
    def _get_local_observation(self, agent_id):
        """获取智能体的本地观察（只能看到当前房间的炸弹情况）"""
        agent_info = self.env.players[agent_id]
        current_room = agent_info['position']
        
        # 只能看到当前房间的炸弹状态
        has_bomb = current_room in self.env.bombs_status and self.env.bombs_status[current_room] > 0
        
        return {
            'position': current_room,
            'has_bomb': has_bomb,
            'adjacent_rooms': list(self.env.adjacency[current_room].keys())
        }
    
    def get_rules(self):
        """获取游戏规则"""
        rules = """
拆弹小队需要在一栋建筑内部进行拆弹工作，需要所有人通过协作完成。
建筑中存在若干房间，房间之间存在通道，玩家可以消耗若干分钟通过通道移动到相邻房间。
你需要通过交流协作完成拆弹任务。
注意：一颗炸弹只能由一个智能体拆除，不能由多个智能体同时拆除.
请注意探索与排查效率，尽量分散探索，避免等待，避免反复探查已探明房间。
"""
        return rules
    
    def get_actions_instruction_text(self, mode='multi'):
        """获取动作执行信息，支持单智能体和多智能体模式"""
        if mode == 'single':
            instructions = """
你现在是唯一的空闲智能体，需要根据当前环境推理，分享信息，并为自己制定行动计划。
如存在[FEEDBACK]，请根据反馈修改计划。

请分析当前情况并为自己规划动作，格式如下：
PLAN
NAME {your_name} ACTION {action}

动作类型：
- WAIT {time}：等待time分钟（1-10）
- MOVE {room}：移动到编号为{room}的相邻房间
- Disposal：拆除当前房间的炸弹（需要5分钟）

请确保你的计划可行且高效。
"""
        else:  # multi mode
            instructions = """
你需要根据信息进行简单的推理，并伙伴分享信息。如存在[FEEDBACK]，请根据反馈修改计划。

当所有智能体至少发言一遍时，可以总结制定行动计划，格式如下：
先输出'EXECUTE'，然后输出 NAME {agent} ACTION {action}为每个空闲智能体规划动作。
例如：
EXECUTE
NAME Alice ACTION MOVE 5
NAME Bob ACTION Disposal
...

动作类型：
- WAIT {time}：等待time分钟（1-10）
- MOVE {room}：移动到编号为{room}的相邻房间
- Disposal：拆除当前房间的炸弹（需要5分钟）

注意：合作规划时禁止采用PLAN格式，否则不会执行PLAN命令。
"""
        return instructions
    
    def get_agents_info_text(self):
        """获取智能体信息"""
        agents_text = []
        for agent_id, name in self.agent_names.items():
            agent_info = self.env.players[agent_id]
            current_room = agent_info['position']
            agents_text.append(f"- {name}: 当前位置为房间{current_room}")
        return "\n".join(agents_text)
    
    def get_local_obs_str(self, agent_id):
        """获取智能体的本地观察信息字符串"""
        local_obs = self._get_local_observation(agent_id)
        current_room = local_obs['position']
        
        # 获取同房间的其他智能体
        same_room_agents = []
        for other_agent_id, other_agent_info in self.env.players.items():
            if other_agent_id != agent_id and other_agent_info['position'] == current_room:
                same_room_agents.append(self.agent_names[other_agent_id])
        
        # 获取当前状态信息
        current_time = self.current_time
        time_limit = self.env.time_limit
        removed_bombs = sum(1 for status in self.env.bombs_status.values() if status == 0)
        total_bombs = len(self.env.bombs_status)
        
        # 获取房间炸弹状态
        if current_room in self.env.bombs_status:
            bomb_status = self.env.bombs_status[current_room]
            if bomb_status <= 0:
                room_status = "炸弹已拆除"
            elif current_room in self.env.defusing_agents:
                # 检查是否有智能体正在拆弹
                defusing_agent_id = self.env.defusing_agents[current_room]
                # 从智能体信息中获取拆弹时间
                defusing_time = self.env.players.get(defusing_agent_id, {}).get('defusing_time', 0)
                room_status = f"炸弹拆除中（剩余{defusing_time}分钟）"
            else:
                room_status = "存在炸弹，未拆除"
        else:
            room_status = "安全"
        
        return f"""当前信息：
- 你的位置：房间{current_room}
- 观察信息：{room_status}
- 同房间智能体：{', '.join(same_room_agents) if same_room_agents else '无'}
- 相邻房间：{', '.join(map(str, local_obs['adjacent_rooms'])) if local_obs['adjacent_rooms'] else '无'}
- 当前时间：{current_time}/{time_limit}
- 炸弹拆除进度：{removed_bombs}/{total_bombs}"""
    
    def get_history_speeches_text(self, k=None):
        """获取历史发言记录文本，支持仅获取最近k轮的发言"""
        if not self.history_speeches:
            return "暂无历史发言"
        
        # 如果指定了k，只获取最近k条发言
        speeches_to_show = self.history_speeches[-k:] if k is not None else self.history_speeches
        history_text = []
        for round_data in speeches_to_show:
            round_num = round_data.get('round', '?')
            for speech in round_data.get('speeches', []):
                history_text.append(f"[轮次：{round_num}（第{speech['time']}分钟）] {speech['speaker']}: {speech['content']}")
        return "\n".join(history_text)
    
    def get_round_speeches_text(self):
        """获取本轮发言记录文本"""
        if self.round_history:
            round_text = []
            for speech in self.round_history:
                round_text.append(f"- {speech['speaker']}: {speech['content']}")
            return "\n".join(round_text)
        else:
            return "本轮暂无发言"
    
    def get_global_adjacency_text(self):
        """获取全局房间联通情况"""
        room_connections = []
        for room in sorted(self.env.adjacency.keys()):
            connections = []
            for neighbor, distance in self.env.adjacency[room].items():
                connections.append(f"房间{neighbor}({distance}分钟)")
            connections_str = ", ".join(connections) if connections else "无"
            room_connections.append(f"房间{room}：{connections_str}")
        return "\n".join(room_connections)
    
    def get_communication_rules_text(self):
        """获取交流规则信息"""
        return """
回复高效，简洁，信息密度大。禁止复述，当可进行总结规划时，必须进行总结规划并输出计划。"""
    
    def get_idle_agents_text(self):
        """获取当前空闲智能体情况"""
        if not self.idle_agents:
            return "当前空闲智能体：无"
        
        idle_agent_names = [self.agent_names[agent_id] for agent_id in self.idle_agents]
        return f"当前空闲智能体：{', '.join(idle_agent_names)}"
    
    def _process_execute_plan(self, response, all_agents_spoken):
        """处理可执行计划
        
        Args:
            response: 智能体的回复
            all_agents_spoken: 是否所有智能体都已发言
            
        Returns:
            tuple: (是否成功, 更新后的响应)
        """
        # 解析并验证计划
        parse_success, actions, parse_feedback = self._parse_action_instruction(response)
        valid_plan = parse_success
        
        for agent_id, action_str in actions.items():
            valid, error_msg = self._validate_action(agent_id, action_str)
            if not valid:
                valid_plan = False
                if parse_feedback:
                    parse_feedback += "；"
                parse_feedback += f"智能体{self.agent_names[agent_id]}的动作无效：{error_msg}"
                break
        
        if valid_plan:
            # 计划成功，作为对话式规划的时间成本，为所有空闲智能体插入一个等待指令
            for agent_id in self.idle_agents:
                self.agent_actions[agent_id]['action_queue'].append(0)  # 等待指令
            
            # 将高级动作转换为低级动作
            self._process_valid_plan(actions)

            return True, response
        else:
            # 计划无效，添加反馈信息
            if parse_feedback:
                response += f"\n\n[FEEDBACK]：{parse_feedback}"
                # 更新上一轮规划失败的反馈信息
                self.last_plan_feedback = parse_feedback
            return False, response
    
    def _reset_roco_state(self):
        """重置RoCo规划的状态"""
        # 将本轮发言添加到历史记录
        if self.round_history:
            # 使用第一条历史的time字段作为本轮的时间
            first_speech_time = self.round_history[0]['time'] if self.round_history else self.current_time
            self.history_speeches.append({
                'round': len(self.history_speeches) + 1,
                'time': first_speech_time,  # 使用第一条历史的time字段
                'speeches': self.round_history.copy()
            })
        
        # 重置状态
        self.round_history = []
        self.current_plan = None
        self.agent_speech_count = None
        self.current_speaker_index = 0
        self.current_communication_count = 0
    
    def _init_roco_state(self):
        """初始化RoCo规划的状态"""
        if not hasattr(self, 'round_history'):
            self.round_history = []
        if not hasattr(self, 'current_plan'):
            self.current_plan = None
        # 只为新加入的智能体设置发言次数为0，保持已有智能体的发言次数不变
        if not hasattr(self, 'agent_speech_count') or self.agent_speech_count is None:
            self.agent_speech_count = {}
        if self.idle_agents:
            for agent_id in self.idle_agents:
                if agent_id not in self.agent_speech_count:
                    self.agent_speech_count[agent_id] = 0
        if not hasattr(self, 'current_communication_count'):
            self.current_communication_count = 0
        if not hasattr(self, 'current_speaker_index'):
            self.current_speaker_index = 0
    
    def _get_prioritized_idle_agents(self):
        """获取优先发言的空闲智能体列表（子类可覆盖）
        
        默认实现返回所有空闲智能体，顺序不变。
        子类如AsynPlanner和CoMAPPlanner可覆盖以实现新智能体优先发言。
        """
        return self.idle_agents
    
    def _get_system_prompt(self, agent_id):
        """获取系统提示，智能体自主选择发言或总结"""
        agent_name = self.agent_names[agent_id]
        
        system_prompt = []
        system_prompt.append(self.get_rules())
        system_prompt.append(f"你是拆弹专家{agent_name}，你只能看到自己当前房间的情况。")
        system_prompt.append(self.get_local_obs_str(agent_id))
        system_prompt.append("\n当前空闲智能体情况：")
        system_prompt.append(self.get_idle_agents_text())
        system_prompt.append("\n智能体状态：")
        system_prompt.append(self.get_agents_status_text())
        
        # 添加上一轮规划失败的反馈信息
        if self.last_plan_feedback:
            system_prompt.append("\n上一轮规划失败的反馈：")
            system_prompt.append(self.last_plan_feedback)
        
        system_prompt.append("\n全局房间联通情况：")
        system_prompt.append(self.get_global_adjacency_text())
        system_prompt.append("\n历史发言记录：")
        system_prompt.append(self.get_history_speeches_text())
        system_prompt.append("\n本轮发言记录：")
        system_prompt.append(self.get_round_speeches_text())
        system_prompt.append(self.get_actions_instruction_text(mode='multi'))
        system_prompt.append(self.get_communication_rules_text())
        
        return '\n'.join(system_prompt)
    
    def _call_language_model(self, system_prompt, user_prompt, agent_name=None, max_retries=3):
        """调用大模型，不需要检测格式正确性"""
        import os
        from openai import OpenAI
        from utils import to_dict

        print(f"system_prompt: {system_prompt}")
        print(f"user_prompt: {user_prompt}")
        
        # 获取模型配置
        model_info = self.model_config['models'].get(self.model_name)
        instruction = None
        usage = None
        
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
                    temperature=model_info.get('temperature', 0.7),
                    max_tokens=model_info.get('max_tokens', 512)
                )
                
                # 提取回答内容
                instruction = response.choices[0].message.content.strip()
                print(f"response: {instruction}")
                
                # 获取token使用统计
                usage = to_dict(response.usage)
                print(usage)
                
                # 更新token使用情况
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    # 根据agent_name获取agent_id
                    agent_id = None
                    if agent_name:
                        for id, name in self.agent_names.items():
                            if name == agent_name:
                                agent_id = id
                                break
                    self.update_token_usage(input_tokens, output_tokens, agent_id)
                
                # 直接返回，不检查格式
                return instruction, usage
                
            except Exception as e:
                print(f"第{attempt + 1}次调用失败：{str(e)}")
        
        return instruction, usage
    
    def _call_model(self, system_prompt, user_prompt, agent_name, speech_count):
        """调用大模型生成回复"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建调用记录
        call_record = {
            'step': self.current_time,
            'timestamp': timestamp,
            'call_count': self.call_count,
            'agent': agent_name,
            'speech_count': speech_count,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
        
        # 保存调用记录
        if self.data_dir:
            # 创建时间步文件夹
            time_dir = os.path.join(self.data_dir, f"step_{self.current_time}")
            os.makedirs(time_dir, exist_ok=True)
            
            filename = f"Agent_{agent_name}_{self.current_time}_{timestamp}_time{speech_count}.json"
            filepath = os.path.join(time_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(call_record, f, indent=2, ensure_ascii=False)
        
        # 模拟大模型回复（实际应用中调用真实模型）
        if self.debug:
            response = self._generate_debug_speech(agent_name)
        else:
            # 使用RoCoPlanner的_call_language_model方法
            response, usage = self._call_language_model(system_prompt, user_prompt, agent_name)
            if response:
                call_record['response'] = response
                call_record['usage'] = usage
            else:
                response = self._generate_debug_speech(agent_name)
        
        # 更新保存的调用记录
        if self.data_dir:
            # 确保保存到时间步文件夹
            time_dir = os.path.join(self.data_dir, f"time_{self.current_time}")
            os.makedirs(time_dir, exist_ok=True)
            
            filename = f"Agent_{agent_name}_{self.current_time}_{timestamp}_time{speech_count}.json"
            filepath = os.path.join(time_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(call_record, f, indent=2, ensure_ascii=False)
        
        return response
    
    def _generate_debug_speech(self, agent_name):
        """生成调试用的模拟发言"""
        agent_id = self._get_agent_id_by_name(agent_name)
        local_obs = self._get_local_observation(agent_id)
        
        if local_obs['has_bomb']:
            return f"我在房间{local_obs['position']}发现了炸弹，需要立即拆除！"
        elif local_obs['adjacent_rooms']:
            target_room = random.choice(local_obs['adjacent_rooms'])
            return f"我在房间{local_obs['position']}，这里没有炸弹，我建议移动到房间{target_room}继续搜索。"
        else:
            return f"我在房间{local_obs['position']}，这里没有炸弹，也没有相邻房间可以移动。"
    
    def _generate_debug_summary(self):
        """生成调试用的模拟总结计划"""
        high_level_actions = {}
        for agent_id in self.idle_agents:
            agent_info = self.env.players[agent_id]
            current_room = agent_info['position']
            
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
        
        return self._convert_action_to_instruction(high_level_actions)
    
    def _execute_roco_planning(self, step, obs):
        """执行RoCo规划流程"""
        # 初始化状态（子类可覆盖）
        self._init_roco_state()

        prioritized_idle_agents = self._get_prioritized_idle_agents()
        
        # 轮流发言直到达到最大交流次数或制定出可行计划
        while self.current_communication_count < self.max_communication_per_minute:
            # 获取当前发言者
            
            if not prioritized_idle_agents:
                break
            
            current_agent_id = prioritized_idle_agents[self.current_speaker_index % len(prioritized_idle_agents)]
            current_agent_name = self.agent_names[current_agent_id]
            
            # 检查该智能体是否已达到最大发言次数
            if self.agent_speech_count[current_agent_id] >= 4:
                self.current_speaker_index += 1
                continue
            
            # 生成回复（智能体自主选择发言或总结）
            system_prompt = self._get_system_prompt(current_agent_id)
            user_prompt = f"你是{current_agent_name}，你的回复："
            response = self._call_model(system_prompt, user_prompt, current_agent_name, self.agent_speech_count[current_agent_id] + 1)
            
            # 更新发言次数
            self.agent_speech_count[current_agent_id] += 1
            self.current_communication_count += 1
            self.call_count += 1
            
            # 检查是否所有空闲智能体都至少发言一次
            all_agents_spoken = all(self.agent_speech_count.get(agent_id, 0) > 0 for agent_id in self.idle_agents)
            
            # 处理反馈信息
            valid = False
            if 'EXECUTE' in response and all_agents_spoken:
                # 处理可执行计划
                valid, updated_response = self._process_execute_plan(response, all_agents_spoken)
                if not valid:
                    response = updated_response
            
            # 将本次发言添加到round_history（所有分支通用）
            self.round_history.append({
                'speaker': current_agent_name,
                'content': response,
                'time': self.current_time
            })
            
            # 如果计划有效，跳出循环
            if valid:
                # 生成EXECUTE格式指令
                parse_success, actions, parse_feedback = self._parse_action_instruction(response)
                if parse_success:
                    instruction_lines = ['EXECUTE']
                    for agent_id, action_str in actions.items():
                        agent_name = self.agent_names[agent_id]
                        instruction_lines.append(f"NAME {agent_name} ACTION {action_str}")
                    plan_instruction = '\n'.join(instruction_lines)
                    
                    # 添加执行计划到round_history
                    self.round_history.append({
                        'speaker': "结果",
                        'content': f"执行计划：{plan_instruction}",
                        'time': self.current_time
                    })
                break
            
            # 切换到下一个发言者
            self.current_speaker_index += 1
        
        # 如果没有制定出可行计划，所有智能体静止
        if not self.current_plan:
            for agent_id in self.idle_agents:
                self.agent_actions[agent_id]['action_queue'].append(0) 

        # 只有在成功规划或达到最大交流次数时才重置状态
        if self.current_plan or self.current_communication_count >= self.max_communication_per_minute:
            self._reset_roco_state()

    def _generate_debug_roco_plan(self):
        """Debug模式：生成模拟的RoCo规划"""
        # 简单模拟RoCo流程
        for agent_id in self.idle_agents:
            agent_info = self.env.players[agent_id]
            current_room = agent_info['position']
            
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
            
            low_level_actions = self._convert_high_level_to_low_level(agent_id, action_str)
            self.agent_actions[agent_id]['action_queue'].extend(low_level_actions)
            self.agent_actions[agent_id]['high_level_actions'].append(action_str)
