import os
import json
import re
import random
from datetime import datetime
from .asyn_planner import AsynPlanner


class CoMAPPlanner(AsynPlanner):
    def __init__(self, env, data_dir=None, debug=False, model_name=None, max_communication_per_minute=6, memory_window=5, k=3):
        super().__init__(env, data_dir, debug, model_name, max_communication_per_minute)
        self.memory = ""  # 共享记忆
        self.memory_history = []  # 记忆历史记录
        self.memory_window = memory_window  # 记忆更新的时间窗口
        self.memory_update_time = None  # 记忆最近更新的时间
        self.consecutive_independent_plans = 0  # 连续独立计划计数
        self.last_plan_type = None  # 上一轮的规划类型 ('independent' 或 'cooperative')
        self.k = k  # 最近k轮的发言记录

    def get_actions(self, obs, step):
        self.current_time = step
        
        # 步骤1: 检测并更新智能体的动作状态
        self._update_agent_action_status()
        
        # 步骤2: 添加空闲智能体到空闲列表
        old_idle_agents = self.idle_agents.copy()
        self.idle_agents = self._get_idle_agents()
        
        # 识别新加入的空闲智能体
        self.new_idle_agents = [agent_id for agent_id in self.idle_agents if agent_id not in old_idle_agents]
        
        # 步骤3: 检查是否需要更新记忆
        if self.idle_agents and (self.consecutive_independent_plans >= 5 or self.last_plan_type == 'cooperative'):
            self._update_memory(obs)
            self.consecutive_independent_plans = 0
        
        # 步骤4: 根据空闲智能体数量选择规划策略
        if len(self.idle_agents) == 1:
            # 只有一个空闲智能体，执行独自计划
            self._execute_single_agent_planning(step, obs)
            self.last_plan_type = 'independent'
            self.consecutive_independent_plans += 1
        elif len(self.idle_agents) > 1:
            # 多个空闲智能体，执行RoCo策略
            if self.debug:
                self._generate_debug_roco_plan()
            else:
                self._execute_roco_planning(step, obs)
            self.last_plan_type = 'cooperative'
            self.consecutive_independent_plans = 0
        
        # 步骤4: 取出动作并修改状态
        actions = self._get_actions_and_update_status()
        
        # 更新上一次的空闲智能体列表
        self.previous_idle_agents = self.idle_agents.copy()
        
        self._save_action_history(actions)
        
        return actions
    
    def _execute_single_agent_planning(self, step, obs):
        """执行单个智能体的独自计划"""
        if not self.idle_agents:
            return
        
        agent_id = self.idle_agents[0]
        agent_name = self.agent_names[agent_id]
        
        # 生成独自计划，使用CoMAP的system_prompt
        system_prompt = self._get_single_agent_system_prompt(agent_id)
        user_prompt = f"你是{agent_name}，请先对当前现状进行简要推理，分享必要信息，最后输出你的计划："
        
        if self.debug:
            # Debug模式：生成模拟的独自计划
            plan = self._generate_debug_single_agent_plan(agent_id)
        else:
            # 调用大模型生成计划
            plan = self._call_model(system_prompt, user_prompt, agent_name, 1)
        
        # 解析并验证计划
        parse_success, actions, parse_feedback = self._parse_single_agent_plan(plan)
        valid_plan = parse_success
        
        for agent_id, action_str in actions.items():
            valid, error_msg = self._validate_action(agent_id, action_str)
            if not valid:
                valid_plan = False
                if parse_feedback:
                    parse_feedback += "；"
                parse_feedback += f"智能体{self.agent_names[agent_id]}的动作无效：{error_msg}"
                break
        
        # 将本次独立计划添加到round_history
        self.round_history.append({
            'speaker': agent_name,
            'content': plan,
            'time': self.current_time
        })
        
        if valid_plan:
            # 计划成功，将高级动作转换为低级动作
            self._process_valid_plan(actions)
            
            # 生成执行计划指令
            if len(self.idle_agents) == 1:
                # 单个智能体，生成PLAN格式指令
                agent_id = list(actions.keys())[0]
                action_str = actions[agent_id]
                agent_name = self.agent_names[agent_id]
                plan_instruction = f"PLAN\nNAME {agent_name} ACTION {action_str}"
            else:
                # 多个智能体，生成EXECUTE格式指令
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
        
        # 无论结果如何，都将本轮独立计划作为一个整体添加到历史记录
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
        self.agent_speech_count = {agent_id: 0 for agent_id in self.idle_agents}
        self.current_speaker_index = 0
        self.current_communication_count = 0
        
        if not valid_plan:
            # 计划无效，执行静止动作
            self.agent_actions[agent_id]['action_queue'].append(0)
            # 规划失败，更新上一轮规划失败的反馈信息
            if parse_feedback:
                self.last_plan_feedback = parse_feedback
    
    def _execute_roco_planning(self, step, obs):
        """执行RoCo规划流程，优先让新智能体发言"""
        # 初始化状态（子类可覆盖）
        self._init_roco_state()

        prioritized_idle_agents=self._get_prioritized_idle_agents()
        
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
            
            # 生成回复（智能体自主选择发言或总结），使用CoMAP的system_prompt
            system_prompt = self._get_system_prompt(current_agent_id)
            user_prompt = f"你是{current_agent_name}，你的回复："
            response = self._call_model(system_prompt, user_prompt, current_agent_name, self.agent_speech_count[current_agent_id] + 1)
            
            # 更新发言次数
            self.agent_speech_count[current_agent_id] += 1
            self.current_communication_count += 1
            self.call_count += 1
            
            # 检查是否所有空闲智能体都至少发言一次
            all_agents_spoken = all(self.agent_speech_count.get(agent_id, 0) > 0 for agent_id in self.idle_agents)
            
            # 检查是否是总结计划（包含EXECUTE标记且所有智能体均已发言）
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
            
            # 如果计划有效，添加执行计划到round_history
            if valid:
                # 解析actions并生成执行计划指令
                parse_success, actions, parse_feedback = self._parse_action_instruction(response)
                if parse_success:
                    # 生成EXECUTE格式指令
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
                # 跳出循环
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
    
    def _get_single_agent_system_prompt(self, agent_id):
        """获取单个智能体独自计划的系统提示，包含记忆"""
        agent_name = self.agent_names[agent_id]
        
        system_prompt = []
        system_prompt.append(self.get_rules())
        system_prompt.append(f"你是拆弹专家{agent_name}，你只能看到自己当前房间的情况。")
        
        # 添加记忆
        if self.memory:
            memory_time_str = f"（更新时间：{self.memory_update_time}分钟）" if self.memory_update_time is not None else ""
            system_prompt.append(f"\n记忆{memory_time_str}：")
            system_prompt.append(self.memory)
        
        system_prompt.append(self.get_local_obs_str(agent_id))
        system_prompt.append("\n智能体状态：")
        system_prompt.append(self.get_agents_status_text())
        
        # 添加上一轮规划失败的反馈信息
        if self.last_plan_feedback:
            system_prompt.append("\n上一轮规划失败的反馈：")
            system_prompt.append(self.last_plan_feedback)
        
        system_prompt.append("\n全局房间联通情况：")
        system_prompt.append(self.get_global_adjacency_text())
        system_prompt.append("\n历史发言记录：")
        system_prompt.append(self.get_history_speeches_text(k=self.k))
        system_prompt.append(self.get_actions_instruction_text(mode='single'))
        
        return '\n'.join(system_prompt)
    
    def get_agents_status_text(self):
        """获取所有智能体当前动作状态的文本描述"""
        status_text = []
        for agent_id, agent_info in self.env.players.items():
            agent_name = self.agent_names[agent_id]
            current_room = agent_info['position']
            status = agent_info['status']
            defusing_time = agent_info.get('defusing_time', 0)
            target_room = agent_info.get('target_room')
            distance_to_target = agent_info.get('distance_to_target', 0)
            
            # 检查智能体是否有高级动作信息
            if agent_id in self.agent_actions:
                current_high_level = self.agent_actions[agent_id].get('current_high_level')
                
                # 检查是否有正在进行的高级动作
                if current_high_level:
                    # 检查是否是移动动作
                    if current_high_level.startswith('MOVE'):
                        # 使用正则表达式解析移动动作，提取目标房间
                        import re
                        match = re.search(r'MOVE\s+(\d+)', current_high_level)
                        if match:
                            move_to_room = match.group(1)
                            # 检查智能体是否已经到达目标房间
                            if str(current_room) == move_to_room:
                                # 到达目标房间，但current_high_level的更新应由_update_agent_action_status处理
                                status_str = f"{agent_name}：在房间{current_room}"
                            else:
                                # 查找移动距离
                                move_distance = self.env.adjacency.get(current_room, {}).get(int(move_to_room), 1)
                                status_str = f"{agent_name}：从房间{current_room}前往房间{move_to_room}(还需{move_distance}分钟)"
                        else:
                            # 动作格式不正确，使用默认状态
                            if defusing_time > 0:
                                status_str = f"{agent_name}：在房间{current_room}拆弹中(还需{defusing_time}分钟)"
                            elif status == 'corridor' and target_room:
                                status_str = f"{agent_name}：从房间{current_room}前往房间{target_room}(还需{distance_to_target}分钟)"
                            else:
                                status_str = f"{agent_name}：在房间{current_room}"
                    elif current_high_level == 'Disposal':
                        # 拆弹动作
                        if defusing_time > 0:
                            status_str = f"{agent_name}：在房间{current_room}拆弹中(还需{defusing_time}分钟)"
                        else:
                            # 拆弹完成，但current_high_level的更新应由_update_agent_action_status处理
                            status_str = f"{agent_name}：在房间{current_room}拆弹完成"
                    elif current_high_level.startswith('WAIT'):
                        # 等待动作
                        parts = current_high_level.split()
                        wait_time = parts[1] if len(parts) == 2 else '1'
                        # 计算剩余等待时间：action_queue中等待动作的数量
                        remaining_waits = sum(1 for a in self.agent_actions[agent_id].get('action_queue', []) if a == 0)
                        status_str = f"{agent_name}：在房间{current_room}等待{wait_time}分钟（还有{remaining_waits}分钟）"
                    else:
                        # 其他动作，使用默认状态
                        if defusing_time > 0:
                            status_str = f"{agent_name}：在房间{current_room}拆弹中(还需{defusing_time}分钟)"
                        elif status == 'corridor' and target_room:
                            status_str = f"{agent_name}：从房间{current_room}前往房间{target_room}(还需{distance_to_target}分钟)"
                        else:
                            status_str = f"{agent_name}：在房间{current_room}"
                else:
                    # 无高级动作信息，使用默认状态
                    if defusing_time > 0:
                        status_str = f"{agent_name}：在房间{current_room}拆弹中(还需{defusing_time}分钟)"
                    elif status == 'corridor' and target_room:
                        status_str = f"{agent_name}：从房间{current_room}前往房间{target_room}(还需{distance_to_target}分钟)"
                    else:
                        status_str = f"{agent_name}：在房间{current_room}"
            else:
                # 无动作信息，使用默认状态
                if defusing_time > 0:
                    status_str = f"{agent_name}：在房间{current_room}拆弹中(还需{defusing_time}分钟)"
                elif status == 'corridor' and target_room:
                    status_str = f"{agent_name}：从房间{current_room}前往房间{target_room}(还需{distance_to_target}分钟)"
                else:
                    status_str = f"{agent_name}：在房间{current_room}"
            status_text.append(status_str)
        
        return '\n'.join(status_text)
    
    def _get_system_prompt(self, agent_id):
        """获取系统提示，智能体自主选择发言或总结，包含记忆"""
        agent_name = self.agent_names[agent_id]
        
        system_prompt = []
        system_prompt.append(self.get_rules())
        system_prompt.append(f"你是拆弹专家{agent_name}，你只能看到自己当前房间的情况。")
        
        # 添加记忆
        if self.memory:
            memory_time_str = f"（更新时间：{self.memory_update_time}分钟）" if self.memory_update_time is not None else ""
            system_prompt.append(f"\n记忆{memory_time_str}：")
            system_prompt.append(self.memory)
        
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
        system_prompt.append("\n历史发言记录（最近{k}轮）：")
        system_prompt.append(self.get_history_speeches_text(k=self.k))
        system_prompt.append("\n本轮发言记录：")
        system_prompt.append(self.get_round_speeches_text())
        system_prompt.append(self.get_actions_instruction_text(mode='multi'))
        system_prompt.append(self.get_communication_rules_text())
        
        return '\n'.join(system_prompt)
    
    def _call_memory_language_model(self, system_prompt, user_prompt, max_retries=3):
        """调用记忆更新专用的大模型"""
        import os
        from openai import OpenAI
        from utils import to_dict

        print(f"memory_system_prompt: {system_prompt}")
        print(f"memory_user_prompt: {user_prompt}")
        
        # 获取记忆模型配置
        memory_model_name = self.model_config.get('memory_model', 'memory')
        model_info = self.model_config['models'].get(memory_model_name)
        model_name=model_info.get('model', 'deepseek-chat')
        memory_content = None
        usage = None

        if "qwen3" in model_name or "Qwen3" in model_name:
            user_prompt += "/no_think"
        
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
                    model=model_info.get('model', 'deepseek-chat'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=model_info.get('temperature', 0.1),
                    max_tokens=model_info.get('max_tokens', 512),
                )
                
                # 提取回答内容
                memory_content = response.choices[0].message.content.strip()
                if "qwen3" in model_name or "Qwen3" in model_name:
                    memory_content = memory_content.split("</think>")[1].strip()

                print(f"memory_response: {memory_content}")
                
                # 获取token使用统计
                usage = to_dict(response.usage)
                print(usage)
                
                # 更新token使用情况
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    self.update_token_usage(input_tokens, output_tokens, None)  # 记忆更新不归属特定智能体
                
                break
            except Exception as e:
                print(f"调用记忆模型失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回空内容
                    memory_content = ""
        
        return memory_content, usage

    def _update_memory(self, obs):
        """更新共享记忆"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备记忆更新的提示
        system_prompt = """你是拆弹小队的记忆整理员，负责整理和总结队伍的任务进展和状态。

请根据旧记忆、环境信息与历史对话信息，更新一段简洁的共享记忆，包括：
1. 任务信息：进度信息。简短，仅使用数字表示。
2. 智能体信息：各个智能体状态。
3. 互动信息：智能体达成的共识、协议规划等。

EXAMPLE：'
[任务信息]
- 探明房间：1;
- 未探房间：2,3;
[智能体状态]
...
[互动信息]
...
'

记忆应该简洁明了，重点突出当前任务的关键信息。
注意，每次更新都要更新房间状态，仅当智能体报告房间信息才能说明该房间已探明，禁止复述上轮记忆内容或猜想臆测。"""
        
        # 准备历史信息
        history_info = []

        history_info.append(f"当前时间：第{self.current_time}分钟")
        
        # 添加上一记忆
        if self.memory:
            history_info.append("上一记忆：")
            history_info.append(self.memory)
        
        # 添加智能体信息
        history_info.append("\n智能体信息：")
        history_info.append(self.get_agents_status_text())
        
        # 添加房间联通情况
        history_info.append("\n房间联通情况：")
        for room in sorted(self.env.adjacency.keys()):
            connections = []
            for neighbor, distance in self.env.adjacency[room].items():
                connections.append(f"房间{neighbor}({distance}分钟)")
            connections_str = ", ".join(connections) if connections else "无"
            history_info.append(f"房间{room}：{connections_str}")
        
        # 添加部分历史对话（从上一次记忆生成时刻开始截取到当前轮次对话）
        if self.history_speeches:
            history_info.append("\n历史对话：")
            # 使用get_history_speeches_text方法获取历史对话
            # for_memory_update=True 表示从最近一次更新的时间开始
            history_text = self.get_history_speeches_text(for_memory_update=True)
            if history_text != "暂无历史发言":
                history_info.append(history_text)
        
        user_prompt = '\n'.join(history_info)
        
        # 调用大模型生成记忆
        if self.debug:
            # 生成模拟记忆
            memory_content = self._generate_debug_memory()
        else:
            # 调用真实模型
            memory_content, usage = self._call_memory_language_model(system_prompt, user_prompt)
            
            # 更新记忆token使用情况
            if usage:
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                self.update_memory_token_usage(input_tokens, output_tokens)
            # if not memory_content:
            #     memory_content = self._generate_debug_memory()
        
        # 更新记忆
        self.memory = memory_content
        # 更新记忆更新时间
        self.memory_update_time = self.current_time
        
        # 保存记忆历史
        memory_record = {
            'timestamp': timestamp,
            'time_step': self.current_time,
            'memory': memory_content,
            'consecutive_independent_plans': self.consecutive_independent_plans,
            'last_plan_type': self.last_plan_type
        }
        self.memory_history.append(memory_record)
        
        # 保存记忆记录到文件
        if self.data_dir:
            memory_filename = f"CoMAP_Memory_{timestamp}.json"
            memory_filepath = os.path.join(self.data_dir, memory_filename)
            with open(memory_filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_record, f, indent=2, ensure_ascii=False)
    
    def _generate_debug_memory(self):
        """生成调试用的模拟记忆"""
        # 生成炸弹进度
        removed_bombs = sum(1 for status in self.env.bombs_status.values() if status == 0)
        total_bombs = len(self.env.bombs_status)
        
        # 生成智能体状态
        agent_statuses = []
        for agent_id, agent_info in self.env.players.items():
            agent_name = self.agent_names[agent_id]
            current_room = agent_info['position']
            defusing_time = agent_info.get('defusing_time', 0)
            
            if defusing_time > 0:
                status = f"{agent_name}在房间{current_room}拆弹（剩余{defusing_time}分钟）"
            else:
                status = f"{agent_name}在房间{current_room}"
            agent_statuses.append(status)
        
        # 生成记忆内容
        memory = f"""任务信息：
- 目标：拆除所有炸弹
- 进度：已拆除{removed_bombs}/{total_bombs}个炸弹
- 探查房间：正在进行中

智能体信息：
- {'; '.join(agent_statuses)}

互动信息：
- 智能体正在协作完成拆弹任务
- 优先处理有炸弹的房间
"""
        
        return memory
    
    def get_memory(self):
        """获取当前共享记忆"""
        return self.memory
    
    def get_memory_history(self):
        """获取记忆历史记录"""
        return self.memory_history
    
    def get_history_speeches_text(self, k=None, for_memory_update=False):
        """获取历史发言记录文本
        
        Args:
            k: 可选，只获取最近k轮的发言
            for_memory_update: 布尔值，是否用于记忆更新
                              - True: 从最近一次更新的时间开始
                              - False: 从最近一次更新记忆时间的前三轮对话开始（默认）
        """
        if not self.history_speeches:
            return "暂无历史发言"
        
        # 找到上一次更新记忆的时间点
        start_index = 0
        if self.memory_history:
            # 获取最近一次记忆更新的时间步
            last_memory_update = self.memory_history[-1]['time_step']
            
            # 找到该时间点对应的发言索引
            for i, round_data in enumerate(self.history_speeches):
                if round_data.get('time', 0) >= last_memory_update:
                    break
                start_index = i
            
            # 用于对话时，从上一次记忆更新时间点前三轮对话开始
            if not for_memory_update:
                start_index = max(0, start_index - 3)
        
        # 获取从start_index开始的发言
        speeches_to_show = self.history_speeches[start_index:]
        
        # 如果指定了k，只获取最近k条发言
        if k is not None and len(speeches_to_show) > k:
            speeches_to_show = speeches_to_show[-k:]
        
        # 生成历史发言文本
        history_text = []
        for round_data in speeches_to_show:
            round_num = round_data.get('round', '?')
            for speech in round_data.get('speeches', []):
                history_text.append(f"[轮次：{round_num}（第{speech['time']}分钟）] {speech['speaker']}: {speech['content']}")
        
        return "\n".join(history_text)