import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


def calculate_identity(seq1, seq2):
    """
    计算两个蛋白质序列之间的identity，填充较短的序列。

    参数:
    seq1 (str): 第一个蛋白质序列
    seq2 (str): 第二个蛋白质序列

    返回:
    float: 序列之间的identity百分比
    """
    min_length = min(len(seq1), len(seq2))
    seq1 = seq1[:min_length]
    seq2 = seq2[:min_length]
    identical_count = sum(1 for a, b in zip(seq1, seq2) if a == b)
    identity_percentage = (identical_count / min_length)
    return identity_percentage


def calculate_identities(generated,val_seqs):
    """
    计算所有序列之间的identity。

    参数:
    sequences (list of str): 蛋白质序列列表

    返回:
    list of float: 所有序列之间的identity百分比
    """
    identities = []
    num_sequences = len(generated)

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            identity = calculate_identity(generated[i], val_seqs[j])
            identities.append(identity)

    return identities


def save_identities_to_csv(identity,step):
    """
    将identity数据保存到CSV文件。

    参数:
    identities_group1 (list of float): 第一组序列之间的identity百分比
    identities_group2 (list of float): 第二组序列之间的identity百分比
    filename (str): CSV文件名
    """
    df = pd.DataFrame([[step],[identity]])  # 将列表数据转换为DataFrame的一行
    df.to_csv("identity.csv", index=False, header=False,mode='a',sep=',')  # 不写入索引和表头





