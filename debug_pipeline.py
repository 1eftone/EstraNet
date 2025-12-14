import h5py
import numpy as np
import os
import subprocess
import sys

# --- 配置部分 ---
DEBUG_FILENAME = "debug_ascadv2_100.h5"
INPUT_LENGTH = 15000    # 模拟 ASCADv2 的长度
NUM_TRACES = 100        # 只用 100 条 Trace
DESYNC = 50             # 模拟一点 Desync
TOTAL_LENGTH = INPUT_LENGTH + DESYNC 

def create_dummy_dataset():
    """创建一个符合 ASCADv2 结构的微型 H5 数据集"""
    print(f"[*] 正在生成微型测试数据集: {DEBUG_FILENAME} ...")
    
    # 1. 生成随机 Trace 数据 (float32)
    # Shape: (100, 15050)
    traces = np.random.randn(NUM_TRACES, TOTAL_LENGTH).astype(np.float32)
    
    # 2. 生成 Metadata (模拟 Plaintext 和 Key)
    # ASCAD 的 metadata 通常是结构化数组
    dt = np.dtype([
        ('plaintext', 'u1', (16,)), 
        ('key', 'u1', (16,)),
        ('ciphertext', 'u1', (16,)),
        ('masks', 'u1', (16,)) # ASCADv1 有 masks，v2 可能结构不同，但这通常不影响我们自己算的 Label
    ])
    
    metadata = np.zeros((NUM_TRACES,), dtype=dt)
    
    # 填充随机 Plaintext 和 Key
    for i in range(NUM_TRACES):
        metadata[i]['plaintext'] = np.random.randint(0, 256, 16, dtype='uint8')
        metadata[i]['key'] = np.random.randint(0, 256, 16, dtype='uint8')
    
    # 3. 写入 H5 文件
    # 我们需要 Profiling_traces 和 Attack_traces 两个组
    with h5py.File(DEBUG_FILENAME, "w") as f:
        # 创建 Profiling_traces 组
        prof_grp = f.create_group("Profiling_traces")
        prof_grp.create_dataset("traces", data=traces)
        prof_grp.create_dataset("metadata", data=metadata)
        # 如果你的 data_utils 依赖 labels 字段，也可以在这里写入，但我们建议动态计算
        # prof_grp.create_dataset("labels", data=...) 
        
        # 创建 Attack_traces 组 (为了通过 test_data 的加载检查，我们也放一点数据)
        attack_grp = f.create_group("Attack_traces")
        attack_grp.create_dataset("traces", data=traces[:10]) # 只要 10 条
        attack_grp.create_dataset("metadata", data=metadata[:10])

    print(f"[+] 数据集生成完毕: {os.path.abspath(DEBUG_FILENAME)}")
    print(f"    - Traces Shape: {traces.shape}")

def run_debug_training():
    """调用 train_trans.py 进行极短的训练"""
    print("\n[*] 开始运行 EstraNet 训练流程测试...")
    
    # 构建命令
    # 我们把步数设得很小，Batch Size 也很小，确保能快速跑完
    cmd = [
        sys.executable, "train_trans.py",
        "--use_tpu=False",
        f"--data_path={DEBUG_FILENAME}", # 指向刚才生成的假文件
        "--dataset=ASCADv2",             # 触发我们修改后的 ASCADv2 逻辑
        "--result_path=results_debug",
        "--checkpoint_dir=./checkpoints_debug",
        
        # --- 关键的 Debug 参数 ---
        f"--input_length={INPUT_LENGTH}",
        "--data_desync=0",               # Debug 时先不加 Desync 也没关系
        "--train_batch_size=8",          # 小 Batch
        "--eval_batch_size=8",
        "--train_steps=5",               # 只跑 5 步！
        "--warmup_steps=0",              # 无需 Warmup
        "--save_steps=5",                # 第 5 步尝试保存模型
        "--iterations=1",                # 每次只迭代 1 次
        "--max_eval_batch=1",            # 验证集也只跑一点点
        
        # --- 模型参数 (保持你的 ASCADv2 配置) ---
        "--beta_hat_2=225",
        "--pool_size=10",
        "--d_kernel_map=512",
        "--do_train=True"
    ]
    
    # 打印命令方便复制
    print(" ".join(cmd))
    print("-" * 50)
    
    # 执行
    # 使用 subprocess 实时输出日志
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    
    if process.returncode == 0:
        print("\n[SUCCESS] 测试通过！流程可以跑通，没有内存溢出。")
        print("现在你可以放心地去跑全量数据了。")
    else:
        print(f"\n[FAIL] 测试失败，返回码: {process.returncode}")
        print("请检查上方的错误日志。")

if __name__ == "__main__":
    # 1. 先生成假数据
    create_dummy_dataset()
    
    # 2. 清理旧的 debug checkpoint (可选)
    # if os.path.exists("./checkpoints_debug"):
    #     import shutil
    #     shutil.rmtree("./checkpoints_debug")
        
    # 3. 运行测试
    run_debug_training()
    
    # 4. 结束后可以删除假数据
    # os.remove(DEBUG_FILENAME)