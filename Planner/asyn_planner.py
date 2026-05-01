import os
import json
import re
import random
from datetime import datetime
from .roco_planner import RoCoPlanner


class AsynPlanner(RoCoPlanner):
    def __init__(self, env, data_dir=None, debug=False, model_name=None, max_communication_per_minute=6):
        super().__init__(env, data_dir, debug, model_name, max_communication_per_minute)
        self.previous_idle_agents = []  # 上一次的空闲智能体列表
        self.new_idle_agents = []  # 新加入的空闲智能体
    
    def get_actions(self, obs, step):
        self.current_time = step
        
        # 步骤1: 检测并更新智能体的动作状态
        self._update_agent_action_status()
        
        # 步骤2: 添加空闲智能体到空闲列表
        old_idle_agents = self.idle_agents.copy()
        self.idle_agents = self._get_idle_agents()
        
        # 识别新加入的空闲智能体
        self.new_idle_agents = [agent_id for agent_id in self.idle_agents if agent_id not in old_idle_agents]
        
        # 步骤3: 根据空闲智能体数量选择规划策略
        if len(self.idle_agents) == 1:
            # 只有一个空闲智能体，执行独自计划
            self._execute_single_agent_planning(step, obs)
        elif len(self.idle_agents) > 1:
            # 多个空闲智能体，执行RoCo策略
            if self.debug:
                self._generate_debug_roco_plan()
            else:
                self._execute_roco_planning(step, obs)
        
        # 步骤4: 取出动作并修改状态
        actions = self._get_actions_and_update_status()
        
        # 更新上一次的空闲智能体列表
        self.previous_idle_agents = self.idle_agents.copy()
        
        self._save_action_history(actions)
        
        return actions
    

    def _get_single_agent_system_prompt(self, agent_id):
        """获取单个智能体独自计划的系统提示"""
        agent_name = self.agent_names[agent_id]
        
        system_prompt = []
        system_prompt.append(self.get_rules())
        system_prompt.append(f"你是拆弹专家{agent_name}，你只能看到自己当前房间的情况。")
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
        system_prompt.append(self.get_history_speeches_text())
        system_prompt.append(self.get_actions_instruction_text(mode='singe'))
        return '\n'.join(system_prompt)
    
    def _parse_single_agent_plan(self, plan):
        """解析单个智能体的计划，返回(解析结果, 解析动作, 反馈信息)"""
        actions = {}
        feedback = ""
        success = True
        
        if not plan or 'PLAN' not in plan:
            feedback = "指令格式错误：独立计划时缺少PLAN标记"
            success = False
            return success, actions, feedback
        
        lines = plan.split('\n')
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
                            error_msg = f"智能体{name}被重复安排动作，忽略重复指令"
                            print(f"警告: {error_msg}")
                            feedback += error_msg + "；"
                            success = False
                            continue
                        # 检查是否为非空闲智能体安排动作
                        if agent_id not in self.idle_agents:
                            error_msg = f"智能体{name}不是空闲状态，忽略动作安排"
                            print(f"警告: {error_msg}")
                            feedback += error_msg + "；"
                            success = False
                            continue
                        # 检查是否有多个智能体在同一房间拆弹
                        if action_str == 'Disposal':
                            agent_info = self.env.players[agent_id]
                            current_room = agent_info['position']
                            if current_room in self.env.defusing_agents:
                                error_msg = f"房间{current_room}已有其他智能体正在拆除炸弹"
                                print(f"警告: {error_msg}")
                                feedback += error_msg + "；"
                                success = False
                                continue
                        actions[agent_id] = action_str
        
        # 检查是否有动作被解析
        if not actions:
            feedback = "未解析到有效的动作指令"
            success = False
        
        # 移除末尾的分号
        if feedback.endswith("；"):
            feedback = feedback[:-1]
        
        return success, actions, feedback
    
    def _execute_single_agent_planning(self, step, obs):
        """执行单个智能体的独自计划"""
        if not self.idle_agents:
            return
        
        agent_id = self.idle_agents[0]
        agent_name = self.agent_names[agent_id]
        
        # 生成独自计划
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
            
            # 生成PLAN格式指令
            agent_id = list(actions.keys())[0]
            action_str = actions[agent_id]
            agent_name = self.agent_names[agent_id]
            plan_instruction = f"PLAN\nNAME {agent_name} ACTION {action_str}"
            
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
            self.last_plan_feedback = parse_feedback
    
    def _generate_debug_single_agent_plan(self, agent_id):
        """生成调试用的模拟独自计划"""
        agent_info = self.env.players[agent_id]
        current_room = agent_info['position']
        agent_name = self.agent_names[agent_id]
        
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
        
        return f"PLAN\nNAME {agent_name} ACTION {action_str}"
    
    def _get_prioritized_idle_agents(self):
        """获取优先发言的空闲智能体列表，新智能体优先"""
        if self.new_idle_agents:
            # 重新排序空闲智能体列表，新智能体优先
            return self.new_idle_agents + [agent_id for agent_id in self.idle_agents if agent_id not in self.new_idle_agents]
        else:
            return self.idle_agents
    
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
            
            # 如果计划有效，跳出循环
            if valid:
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
