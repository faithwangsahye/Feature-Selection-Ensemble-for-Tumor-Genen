import pandas as pd
import numpy as np

def Counting(importance_list, num=200):
    """
    统计前 num 个特征在各个算法中的出现次数。

    参数:
    - importance_list: 特征重要性列表
    - num: 需要统计的前 N 个特征 (默认为 200)

    返回值:
    - df_genecounts: 各个基因出现次数的 DataFrame
    - rankremain: 保留的特征排名
    """
    rankremain = importance_list[:num]
    gene_counts = rankremain.stack().value_counts()
    df_genecounts = pd.DataFrame(gene_counts.values, index=[gene_counts.index])
    df_genecounts.rename(columns={df_genecounts.columns[0]: 'Counts'}, inplace=True)
    return df_genecounts, rankremain


def RerankGenes(df_genecounts, rankremain):
    """
    重新为特征进行排名。

    参数:
    - df_genecounts: 各个基因的出现次数
    - rankremain: 保留的特征排名

    返回值:
    - Rerank: 重新排序后的特征
    """
    appeargene = df_genecounts.index.tolist()
    agorithem = rankremain.columns.tolist()
    appeargene = [item[0] for item in appeargene]
    Rerank = pd.DataFrame(index=appeargene, columns=agorithem)
    
    for column in rankremain.columns:
        for index, value in rankremain[column].items():
            if pd.notna(value):
                if pd.isna(Rerank.at[value, column]):
                    Rerank.at[value, column] = str(index)
                else:
                    Rerank.at[value, column] += f', {index}'
    return Rerank


def convert_to_numeric(x):
    """
    尝试将值转换为数值类型，如果无法转换则保留原始值。

    参数:
    - x: 输入值

    返回值:
    - 转换后的数值或原始值
    """
    if pd.isna(x):
        return x
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x


def transform_Scoring(x):
    """
    根据特征排名对其进行评分，排名越靠前分数越高。

    参数:
    - x: 特征排名

    返回值:
    - 转换后的分数
    """
    if pd.isnull(x):
        return 0
    else:
        return 201 - x


def process_data(importance_list):
    """
    对特征重要性列表进行处理，返回重新排序后的特征评分。

    参数:
    - importance_list: 特征重要性列表

    返回值:
    - Scoring: 处理后的特征评分
    """
    df_genecounts, rankremain = Counting(importance_list)
    Rerank = RerankGenes(df_genecounts, rankremain)
    Rerankint = Rerank.applymap(convert_to_numeric)
    Scoring = Rerankint.applymap(transform_Scoring).transpose()
    return Scoring


def scale_order(Scoring, new_min):
    """
    对特征评分进行归一化并重新排序。

    参数:
    - Scoring: 特征评分 DataFrame
    - new_min: 归一化时的最小值

    返回值:
    - order: 重新排序后的特征顺序
    - StandardScore: 标准化后的特征评分
    """
    normalized_columns = []
    for column in Scoring.columns:
        original_min = 1
        original_max = Scoring[column].max()
        new_max = 200
        
        scaled_data = np.where(Scoring[column] == 0, 0,
                               (Scoring[column] - original_min) / (original_max - original_min) * (new_max - new_min) + new_min)
        scaled_series = pd.Series(scaled_data, index=Scoring.index, name=column)
        normalized_columns.append(scaled_series)
        
    StandardScore = pd.concat(normalized_columns, axis=1)
    Avgscore = StandardScore.mean()
    StandardScore.loc['Average'] = Avgscore
    order = Avgscore.sort_values(ascending=False).index
    StandardScore = StandardScore.reindex(columns=order).round(2)
    return order, StandardScore


def Step_in_out_ACC(x_train, y_train, feature_list, numb, cv_n):
    """
    逐步选入和剔除特征，以提升分类准确性。

    参数:
    - x_train: 训练集特征
    - y_train: 训练集标签
    - feature_list: 特征列表
    - numb: 选入特征的数量
    - cv_n: 交叉验证折数

    返回值:
    - current_ACC: 每一步的分类准确性
    - selected_features: 最终选定的特征列表
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=cv_n, random_state=42, shuffle=True)
    current_ACC = [0]
    selected_features = []

    for i in range(numb):
        current_feature = feature_list[i]
        subtrain = x_train[selected_features + [current_feature]]
        svm_model = SVC(probability=True)
        scoring = {'ACC': 'accuracy'}
        results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
        average_acc = results['test_ACC'].mean()

        if current_ACC[-1] < average_acc:
            selected_features.append(current_feature)
            current_ACC.append(average_acc)
    
    return current_ACC, selected_features
