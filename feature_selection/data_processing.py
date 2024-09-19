import pandas as pd
from sklearn.model_selection import train_test_split

def data_import(file_path, label_col='sample', target_col='ylable', n_features=30557, test_size=0.2, random_state=111):
    """
    导入数据，并对数据进行预处理，包括对样本进行划分和特征选择。
    
    参数:
    - file_path: 数据文件的路径 (csv格式)
    - label_col: 样本的标记列名称 (默认为 'sample')
    - target_col: 目标标签列名称 (默认为 'ylable')
    - n_features: 使用的特征数目 (默认为 30557)
    - test_size: 测试集所占比例 (默认为 0.2)
    - random_state: 随机种子，确保结果的可重复性 (默认为 111)
    
    返回值:
    - x: 样本特征 (完整数据)
    - y: 样本标签 (完整数据)
    - data: 训练数据 (划分后的训练集)
    - final: 测试数据 (划分后的测试集)
    - x_data: 训练集的特征
    - y_data: 训练集的标签
    - x_final: 测试集的特征
    - y_final: 测试集的标签
    - gene_name: 基因名称列表
    """
    
    
    exp_data = pd.read_csv(file_path, sep=',')
    exp_data.columns.values[0] = label_col
    gene_name = exp_data[label_col]
    exp_data = exp_data.drop(labels=label_col, axis=1).transpose()
    
    ylable = pd.DataFrame(exp_data.index.str.split('-').str[-1])
    exp_data.columns = gene_name
    
    del gene_name[n_features]
    
    x = exp_data.iloc[:, 0:n_features]
    y = exp_data.iloc[:, -1].astype(int)
    
    # 平衡数据集
    size = min(y.value_counts()[0], y.value_counts()[1])
    normal_data = exp_data[y == 0].sample(n=size)
    cancer_data = exp_data[y == 1].sample(n=size)
    equal_data = pd.concat([normal_data, cancer_data], axis=0)
    
    eq_x = equal_data.iloc[:, 0:n_features]
    eq_y = equal_data.iloc[:, -1].astype(int)
    x_data, x_final, y_data, y_final = train_test_split(eq_x, eq_y, test_size=test_size, random_state=random_state)
    
    data = pd.concat([x_data, y_data], axis=1)
    final = pd.concat([x_final, y_final], axis=1)
    
    return x, y, data, final, x_data, y_data, x_final, y_final, gene_name


def train(file_path):
    """
    从CSV文件中加载训练数据，并进行预处理。
    
    参数:
    - file_path: CSV文件的路径
    
    返回值:
    - x_data: 训练集的特征
    - y_data: 训练集的标签
    """
    data = pd.read_csv(file_path, sep=',')
    data.columns.values[0] = 'sample'
    data.index = data['sample']
    del data['sample']
    
    x_data = pd.DataFrame(data.iloc[:, :-1])
    y_data = data["yabel"]
    
    return x_data, y_data


def final(file_path):
    """
    从CSV文件中加载测试数据，并进行预处理。
    
    参数:
    - file_path: CSV文件的路径
    
    返回值:
    - x_final: 测试集的特征
    - y_final: 测试集的标签
    """
    final = pd.read_csv(file_path, sep=',')
    final.columns.values[0] = 'sample'
    final.index = final['sample']
    del final['sample']
    
    x_final = pd.DataFrame(final.iloc[:, :-1])
    y_final = final["yabel"]
    
    return x_final, y_final


def Counting(importance_list, num=200):
    """
    统计前num个特征在各个算法中的出现次数。
    
    参数:
    - importance_list: 特征重要性列表
    - num: 需要统计的前N个特征 (默认为 200)
    
    返回值:
    - df_genecounts: 各个基因出现次数的DataFrame
    - rankremain: 保留的特征排名
    """
    rankremain = importance_list[:num]
    gene_counts = rankremain.stack().value_counts()
    df_genecounts = pd.DataFrame(gene_counts.values, index=[gene_counts.index])
    df_genecounts.rename(columns={df_genecounts.columns[0]: 'Counts'}, inplace=True)
    
    return df_genecounts, rankremain
