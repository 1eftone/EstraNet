import tensorflow as tf
import numpy as np
import h5py
import os
import sys

# 导入 Transformer 类
try:
    from transformer import Transformer as estranet_model
except ImportError:
    print("错误: 当前目录下找不到 transformer.py，请确保脚本在 EstraNet 根目录运行")
    sys.exit(1)

# --- 配置 ---
DATA_PATH = "/home/e240023/EstraNet/datasets/ascadv2-extracted.h5" 
INPUT_LENGTH = 15000
NUM_TRACES = 50  # 只用 50 条进行过拟合测试
TARGET_BYTE = 1  # 目标字节

# S-box 表
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])

def load_sanity_data():
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 数据文件不存在: {DATA_PATH}")
        sys.exit(1)
        
    print(f"[*] Loading top {NUM_TRACES} traces for sanity check...")
    with h5py.File(DATA_PATH, "r") as f:
        # Trace shape: (50, 15000)
        traces = f['Profiling_traces']['traces'][:NUM_TRACES, :INPUT_LENGTH].astype(np.float32)
        
        # Metadata reading
        # 根据经验，ASCADv2 H5 metadata 可能是 structured array
        try:
            # 尝试直接读取列
            plt = f['Profiling_traces']['metadata']['plaintext'][:NUM_TRACES]
            keys = f['Profiling_traces']['metadata']['key'][:NUM_TRACES]
        except:
            # 如果失败，尝试逐行读取 (兼容性更强)
            print("[Info] Switching to row-by-row metadata reading...")
            meta = f['Profiling_traces']['metadata'][:NUM_TRACES]
            plt = np.array([x['plaintext'] for x in meta])
            keys = np.array([x['key'] for x in meta])
    
    # 1. 归一化 (关键步骤)
    print("[*] Normalizing traces...")
    mean = np.mean(traces, axis=0, keepdims=True)
    std = np.std(traces, axis=0, keepdims=True)
    std[std == 0] = 1.0 
    traces = (traces - mean) / std
    
    # 2. 计算 Label
    print(f"[*] Calculating labels for Byte {TARGET_BYTE}...")
    p_byte = plt[:, TARGET_BYTE]
    k_byte = keys[:, TARGET_BYTE]
    labels = SBOX[p_byte ^ k_byte]
    
    return traces, labels

def main():
    BATCH_SIZE = 10
    
    # 1. 准备数据
    x, y = load_sanity_data()
    # 注意：这里不需要 expand_dims，Transformer 内部会处理
    print(f"[*] Input shape: {x.shape}") 
    
    # Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(BATCH_SIZE).repeat()
    
    # 2. 构建 EstraNet 模型
    print("[*] Building EstraNet (Transformer)...")
    model = estranet_model(
        n_layer=2,
        d_model=128,
        d_head=32,
        n_head=4,
        d_inner=256,
        d_head_softmax=32,
        n_head_softmax=4,
        dropout=0.0,
        n_classes=256,
        conv_kernel_size=3,
        n_conv_layer=1,
        pool_size=4,
        d_kernel_map=128,
        beta_hat_2=150,
        model_normalization='preLC',
        head_initialization='forward',
        softmax_attn=True,
        output_attn=False
    )
    
    # 3. 编译 (关键：run_eagerly=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn, 
        metrics=['accuracy'],
        run_eagerly=True  # <--- 【关键修复】强制 Eager 模式运行，解决源码中的 reshape bug
    )
    
    # 4. 训练
    print("[*] Starting Sanity Check Training (Target: Loss < 0.1)...")
    steps = NUM_TRACES // BATCH_SIZE
    history = model.fit(dataset, steps_per_epoch=steps, epochs=200, verbose=1)
    
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    
    print("-" * 30)
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Acc : {final_acc:.4f}")
    
    if final_loss < 0.1:
        print("✅ [SUCCESS] Sanity Check Passed! 模型成功记住了数据。")
        print("下一步：请在全量训练脚本中，减小 Desync (设为0或50)，并确认使用 Target Byte 1。")
    else:
        print("❌ [FAIL] Sanity Check Failed! Loss 依然很高。")
        print("可能性：Label 计算错误 (Target Byte 不对?) 或者 数据提取窗口无泄露。")

if __name__ == "__main__":
    main()