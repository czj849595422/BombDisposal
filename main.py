import argparse
import os
import json
import time
from datetime import datetime
from email.policy import default

from BombDisposal import BombDisposalEnv
from Planner import BasePlanner, SyncPlanner, OraclePlanner, RoCoPlanner, AsynPlanner, CoMAPPlanner

def save_step_data(data_dir, step, obs, actions, reward, terminated, truncated):
    step_dir = os.path.join(data_dir, f'step_{step}')
    os.makedirs(step_dir, exist_ok=True)
    
    data = {
        'step': step,
        'observation': obs,
        'actions': actions,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated
    }
    
    with open(os.path.join(step_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=2)

def save_game_summary(data_dir, config, results):
    summary = {
        'config': config,
        'results': results
    }
    
    with open(os.path.join(data_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='炸弹拆除游戏主程序')
    parser.add_argument('--task_file', type=str, default='test_task2.json', help='任务配置文件路径')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--num-agents', type=int, default=2, help='智能体数量')
    parser.add_argument('--planner-type', type=str, default='comap', help='Planner类型')
    parser.add_argument('--model', type=str, help='模型名称',default="deepseek")
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--data-dir', type=str, default='./test', help='实验数据保存目录')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 构建子文件夹路径：{planner}_{agent_number}
    planner_folder = f"{args.planner_type}_{args.num_agents}"
    
    # 构建任务名称/种子部分
    if args.task_file:
        # 从任务文件路径中提取任务名称（不含扩展名）
        task_name = os.path.splitext(os.path.basename(args.task_file))[0]
        task_identifier = task_name
    else:
        # 没有任务文件时使用种子
        task_identifier = f"seed_{args.seed}"
    
    # 构建最终的保存目录：{task_identifier}_{timestamp}
    run_folder = f"{task_identifier}_{timestamp}"
    data_dir = os.path.join(args.data_dir, planner_folder, run_folder)
    os.makedirs(data_dir, exist_ok=True)
    
    config = {
        'task_file': args.task_file,
        'seed': args.seed,
        'num_agents': args.num_agents,
        'planner_type': args.planner_type,
        'model': args.model,
        'debug': args.debug,
        'data_dir': args.data_dir,
        'timestamp': timestamp
    }
    
    with open(os.path.join(data_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.task_file:
        env = BombDisposalEnv(task_file=args.task_file, num_players=args.num_agents, seed=args.seed)
    else:
        env = BombDisposalEnv(num_players=args.num_agents, seed=args.seed)
    
    if args.planner_type == 'sync':
        planner = SyncPlanner(env, data_dir)
    elif args.planner_type == 'oracle':
        planner = OraclePlanner(env, data_dir, debug=args.debug, model_name=args.model)
    elif args.planner_type == 'roco':
        planner = RoCoPlanner(env, data_dir, debug=args.debug, model_name=args.model)
    elif args.planner_type == 'asyn':
        planner = AsynPlanner(env, data_dir, debug=args.debug, model_name=args.model)
    elif args.planner_type == 'comap':
        planner = CoMAPPlanner(env, data_dir, debug=args.debug, model_name=args.model)
    else:
        planner = BasePlanner(env, data_dir)
    
    obs, info = env.reset()
    # print(obs['players'])
    # print(obs['bombs'])
    total_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    save_step_data(data_dir, step_count, obs, {}, 0, terminated, truncated)
    planner.save_action(step_count)
    
    while not terminated and not truncated:
        actions = planner.get_actions(obs, step_count)
        print(actions)
        
        obs, reward, terminated, truncated, info = env.step(actions)
        print(obs['players'])
        print(obs['bombs'])

        total_reward += reward
        step_count += 1
        
        save_step_data(data_dir, step_count, obs, actions, reward, terminated, truncated)
        planner.save_action(step_count)
    
    # 检查planner是否有token使用统计
    if hasattr(planner, 'get_token_stats'):
        token_stats = planner.get_token_stats()
        token_stats_text = planner.get_token_stats_text()
        print(token_stats_text)
    else:
        token_stats = None
        token_stats_text = "无token使用统计"
    
    results = {
        'total_steps': step_count,
        'total_reward': total_reward,
        'terminated': terminated,
        'truncated': truncated,
        'explored_rooms': len(env.explored_rooms),
        'total_rooms': env.A,
        'remaining_bombs': sum(1 for status in env.bombs_status.values() if status > 0),
        'total_bombs': env.B,
        'token_stats': token_stats
    }
    
    # 保存token使用情况到文件
    if token_stats:
        with open(os.path.join(data_dir, 'token_stats.json'), 'w') as f:
            json.dump(token_stats, f, indent=2)
        
        with open(os.path.join(data_dir, 'token_stats.txt'), 'w', encoding='utf-8') as f:
            f.write(token_stats_text)
    
    save_game_summary(data_dir, config, results)
    
    print(f"游戏完成！")
    print(f"  数据保存路径: {data_dir}")
    print(f"  总分钟数: {step_count}")
    print(f"  总奖励: {total_reward}")
    print(f"  结果: {'胜利' if terminated else '超时'}")
    print(f"  探索房间: {len(env.explored_rooms)}/{env.A}")
    print(f"  剩余炸弹: {sum(1 for status in env.bombs_status.values() if status > 0)}/{env.B}")

if __name__ == "__main__":
    main()