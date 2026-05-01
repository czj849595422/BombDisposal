# BombDisposal - 炸弹拆除多智能体协作系统

一个基于大语言模型的炸弹拆除多智能体协作研究平台，提供多种规划算法和环境模拟。

## 项目简介

本项目是一个模拟炸弹拆除场景的多智能体协作研究平台，通过大语言模型驱动智能体进行规划和决策。系统提供了从基础随机策略到高级记忆增强协作的多种规划器，支持研究不同协作机制下的智能体行为。

## 目录结构

### 核心文件

| 文件/文件夹 | 说明 |
|------------|------|
| `main.py` | 主程序入口，支持命令行参数配置，运行游戏并保存数据 |
| `model.json` | 大模型配置文件，支持多种模型提供商和参数设置 |
| `environment.yml` | Conda环境配置文件 |
| `utils.py` | 工具函数 |

### 环境模块

| 文件/文件夹 | 说明 |
|------------|------|
| `BombDisposal/` | 游戏环境实现，包含房间布局、炸弹分布、智能体移动等逻辑 |

### 规划器模块

| 文件/文件夹 | 说明 |
|------------|------|
| `Planner/__init__.py` | 规划器模块初始化，导出所有规划器类 |
| `Planner/planner_rules.md` | 各规划器的详细规则和工作机制说明 |
| `Planner/base_planner.py` | BasePlanner - 基础规划器，提供随机策略和状态管理框架 |
| `Planner/sync_planner.py` | SyncPlanner - 同步规划器，所有智能体必须同时空闲才能规划 |
| `Planner/oracle_planner.py` | OraclePlanner - 大模型规划器，直接调用LLM生成动作指令 |
| `Planner/roco_planner.py` | RoCoPlanner - 轮流交流规划器，模拟团队讨论协作 |
| `Planner/asyn_planner.py` | AsynPlanner - 异步规划器，支持单个智能体独自计划和多智能体协作 |
| `Planner/comap_planner.py` | CoMAPPlanner - 记忆增强规划器，引入时间窗口和共享记忆功能 |

### 任务和数据

| 文件/文件夹 | 说明 |
|------------|------|
| `task/` | 任务配置文件夹，包含游戏场景描述和炸弹分布 |
| `task_generator.py` | 任务生成器，用于创建新的游戏任务 |


### 实验工具

| 文件/文件夹 | 说明 |
|------------|------|
| `run_experiments.sh` | 自动化实验脚本，支持并行运行多个任务 |

## 规划器说明

### 1. BasePlanner
提供智能体动作管理的基础框架，使用随机策略，优先选择拆弹动作。

### 2. SyncPlanner
实现同步执行策略，所有智能体必须同时空闲才能分配新动作，确保团队协调。

### 3. OraclePlanner
利用大模型直接生成高级动作指令，支持调试模式和真实模型调用。

### 4. RoCoPlanner
实现智能体轮流发言的交流机制，所有空闲智能体轮流分享信息后进行总结规划。

### 5. AsynPlanner
根据空闲智能体数量采用不同策略：单个智能体独自计划，多个智能体轮流交流。

### 6. CoMAPPlanner
引入时间窗口机制和共享记忆功能，提高长期任务的规划效率。

详细规则请参考 [Planner/planner_rules.md](./Planner/planner_rules.md)

## 快速开始

### 环境安装

```bash
conda env create -f environment.yml
conda activate bomb
```

### 配置模型

编辑 `model.json` 文件，配置你的大模型API：

```json
{
    "models": {
        "deepseek": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_base": "https://api.deepseek.com",
            "api_key": "your_api_key",
            "temperature": 0.7,
            "max_tokens": 512
        }
    },
    "default_model": "deepseek"
}
```

### 运行单次实验

```bash
python main.py --task_file task/my_task.json --num-agents 2 --planner-type comap --model deepseek --data-dir ./data
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--task_file` | 任务配置文件路径 | test_task2.json |
| `--seed` | 随机种子 | None |
| `--num-agents` | 智能体数量 | 2 |
| `--planner-type` | 规划器类型（base/sync/oracle/roco/asyn/comap） | comap |
| `--model` | 模型名称（对应model.json中的配置） | deepseek |
| `--debug` | 启用调试模式（不调用真实模型） | False |
| `--data-dir` | 实验数据保存目录 | ./test |

### 运行批量实验

使用 `run_experiments.sh` 脚本批量运行实验：

```bash
./run_experiments.sh
```


## 开发指南

### 创建自定义任务

1. 复制现有的任务文件或使用 `task_generator.py` 创建新任务
2. 编辑任务文件，定义房间布局、炸弹位置、时间限制等
3. 使用新任务运行实验
