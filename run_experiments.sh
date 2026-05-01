#!/bin/bash

# 炸弹拆除游戏实验自动化脚本
# 用法: ./run_experiments.sh <planner_type> <num_agents> [data_dir]

# 参数检查
if [ $# -lt 2 ]; then
    echo "用法: $0 <planner_type> <num_agents> [data_dir]"
    echo "planner_type: 规划器类型 (base, sync, oracle, roco, asyn, comap)"
    echo "num_agents: 智能体数量"
    echo "data_dir: 数据保存目录 (默认: ./experiment)"
    exit 1
fi

PLANNER_TYPE="$1"
NUM_AGENTS="$2"
DATA_DIR="${3:-./experiment}"

# 任务文件夹
TASK_DIR="./task"

# 最大并行数
MAX_PARALLEL=5

# 创建数据目录
mkdir -p "$DATA_DIR"

# 获取所有任务文件
TASK_FILES=()
while IFS= read -r -d $'\0'; do
    TASK_FILES+=("$REPLY")
done < <(find "$TASK_DIR" -maxdepth 1 -name "*.json" -print0 | sort -z)

# 检查任务数量
NUM_TASKS=${#TASK_FILES[@]}
if [ "$NUM_TASKS" -eq 0 ]; then
    echo "错误: 在 $TASK_DIR 中没有找到任务文件"
    exit 1
fi

echo "========================================"
echo "炸弹拆除实验自动化脚本"
echo "========================================"
echo "规划器类型: $PLANNER_TYPE"
echo "智能体数量: $NUM_AGENTS"
echo "数据保存目录: $DATA_DIR"
echo "任务文件数量: $NUM_TASKS"
echo "最大并行数: $MAX_PARALLEL"
echo "========================================"
echo ""

# 声明一个关联数组来跟踪正在运行的任务
declare -A running_tasks

# 任务计数器
task_counter=0

# 等待函数
wait_for_free_slot() {
    while [ ${#running_tasks[@]} -ge "$MAX_PARALLEL" ]; do
        # 检查哪些任务已经完成
        for pid in "${!running_tasks[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                task_name="${running_tasks[$pid]}"
                wait "$pid" 2>/dev/null
                exit_code=$?
                echo "任务完成: $task_name (退出码: $exit_code)"
                unset "running_tasks[$pid]"
            fi
        done
        # 短暂休眠
        sleep 1
    done
}

# 运行所有任务
for task_file in "${TASK_FILES[@]}"; do
    task_counter=$((task_counter + 1))
    task_name=$(basename "$task_file" .json)
    
    # 等待空闲槽位
    wait_for_free_slot
    
    echo "开始任务 $task_counter/$NUM_TASKS: $task_name"
    
    # 运行任务
    python main.py \
        --task_file "$task_file" \
        --num-agents "$NUM_AGENTS" \
        --planner-type "$PLANNER_TYPE" \
        --data-dir "$DATA_DIR" &
    
    # 记录任务PID
    pid=$!
    running_tasks[$pid]="$task_name"
    echo "任务已启动: $task_name (PID: $pid)"
done

# 等待所有任务完成
echo ""
echo "所有任务已启动，等待剩余任务完成..."
wait

echo ""
echo "========================================"
echo "所有实验已完成！"
echo "========================================"
echo "实验数据保存在: $DATA_DIR"
