import numpy as np
import tensorflow as tf
import h5py
import os, sys

# 1. SBOX 表
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

class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0, target_byte=2):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync
        self.target_byte = target_byte

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'

        # 读取原始数据 (保持原类型，节省内存！)
        # 注意：这里我们读取长度为 input_length + data_desync，以便后续做 shift
        self.traces = corpus[split_key]['traces'][:, :(self.input_length+self.data_desync)]
        
        # 【关键修改】只计算统计量，不修改 self.traces 本身
        print("Calculating mean and std for normalization...")
        self.mean = np.mean(self.traces, axis=0, keepdims=True).astype(np.float32)
        self.std = np.std(self.traces, axis=0, keepdims=True).astype(np.float32)
        self.std[self.std == 0] = 1.0
        print("Stats calculated. (Normalization will apply on-the-fly)")
        # ... 在计算完 mean 和 std 之后 ...
        print(f"[DEBUG] Mean range: {self.mean.min()} ~ {self.mean.max()}")
        print(f"[DEBUG] Std  range: {self.std.min()} ~ {self.std.max()}")
        if self.std.min() < 1e-6:
            print("⚠️ 警告: Std 有极小值，可能会导致梯度爆炸！")

        # 动态获取 Plaintext 和 Key
        self.plaintexts = self.GetPlaintexts(corpus[split_key]['metadata'])
        self.keys = self.GetKeys(corpus[split_key]['metadata'])
        
        # 计算 Label
        print(f"Calculating labels for Target Byte: {self.target_byte}")
        self.labels = SBOX[self.plaintexts ^ self.keys]
        self.labels = np.reshape(self.labels, [-1, 1]).astype(np.int64)
        
        self.num_samples = self.traces.shape[0]

        # 数据切分
        max_split_size = 2000000000//self.input_length
        split_idx = list(range(max_split_size, self.num_samples, max_split_size))
        self.traces = np.split(self.traces, split_idx, axis=0)
        self.labels = np.split(self.labels, split_idx, axis=0)

    def GetPlaintexts(self, metadata):
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]['plaintext'][self.target_byte])
        return np.array(plaintexts)

    def GetKeys(self, metadata):
        keys = []
        for i in range(len(metadata)):
            if 'key' in metadata[i].dtype.names:
                keys.append(metadata[i]['key'][self.target_byte])
            else:
                keys.append(metadata[i][self.target_byte])
        return np.array(keys)

    def GetTFRecords(self, batch_size, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.traces[0], self.labels[0]))
        for traces, labels in zip(self.traces[1:], self.labels[1:]):
            temp_dataset = tf.data.Dataset.from_tensor_slices((traces, labels))
            dataset.concatenate(temp_dataset)

        # 【核心逻辑】先归一化，再 Shift (或者先 Cast 再归一化)
        # 这里的顺序很重要：我们的 mean/std 是基于 (input_length + data_desync) 计算的
        # 所以要在 Shift 之前应用归一化
        def preprocess(x, max_desync):
            # 1. 转换类型 (Int -> Float32)
            x = tf.cast(x, tf.float32)
            # 2. 归一化 (减均值除方差)
            x = (x - self.mean) / self.std
            
            # 3. 数据增强 (Shift/Crop)
            ds = tf.random.uniform([1], 0, max_desync+1, tf.dtypes.int32)
            ds = tf.concat([[0], ds], 0) # 补全维度
            x = tf.slice(x, ds, [-1, self.input_length]) # 裁剪到 input_length
            return x

        # 4. 验证集不做 Desync，但要裁剪
        def preprocess_test(x):
            x = tf.cast(x, tf.float32)
            x = (x - self.mean) / self.std
            ds = tf.constant([0, 0], dtype=tf.int32) # 不偏移
            x = tf.slice(x, ds, [-1, self.input_length])
            return x

        if training == True:
            # 训练集 pipeline
            return dataset.repeat() \
                          .shuffle(self.num_samples) \
                          .batch(batch_size//2) \
                          .map(lambda x, y: (preprocess(x, self.data_desync), y)) \
                          .unbatch() \
                          .batch(batch_size, drop_remainder=True) \
                          .prefetch(10)
        else:
            # 测试集 pipeline
            return dataset.batch(batch_size, drop_remainder=True) \
                          .map(lambda x, y: (preprocess_test(x), y)) \
                          .prefetch(10)

    def GetDataset(self):
        return self.traces, self.labels

if __name__ == '__main__':
    # 简单测试代码
    pass