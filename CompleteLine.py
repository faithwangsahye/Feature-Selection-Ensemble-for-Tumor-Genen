'''
Created on 2023/12/7

@author: HuangSihui
'''

import csv, sys, os
#import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import random

workdir = os.path.abspath('.')

# === 除去相似的排序 ===


# === Counting ===

ImportanceRanking = pd.read_csv('C:/Users/14816/Desktop/Workdir/Done/ImportanceRanking.csv',sep=',',index_col = 0)[:20530]
#取排名前200，统计基因在9个算法重要性排名前200中出现的次数，区间[1,9]
rankremain = ImportanceRanking[:200]

gene_counts = rankremain.stack().value_counts()
# 打印元素的出现次数
#for gene, count in gene_counts.items():
#    print(f"Gene {gene} count: {count}")
df_genecounts = pd.DataFrame(gene_counts.values,index = [gene_counts.index] )
df_genecounts.rename(columns={df_genecounts.columns[0]: 'Counts'}, inplace=True)
#rankremain.stack()[1] #check


# === Rerank ===

appeargene = df_genecounts.index.tolist()
agorithem = rankremain.columns.tolist()
appeargene = [item[0] for item in appeargene]
Rerank = pd.DataFrame(index=appeargene, columns=agorithem)

for column in rankremain.columns:
    for index, value in rankremain[column].items():
        if pd.notna(value):  # 检查非NA值
            if pd.isna(Rerank.at[value, column]):
                Rerank.at[value, column] = str(index)
            else:
                Rerank.at[value, column] += f', {index}'
                

# === Scoring ===

## 数值型格式转换
def convert_to_numeric(x):
    if pd.isna(x):
        return x  # 如果是NaN，保留原值
    try:
        # 尝试将字符串或浮点数转换为数值型
        return pd.to_numeric(x)
    except ValueError:
        return x  # 如果转换失败，保留原值
    
Rerankint = Rerank.applymap(convert_to_numeric)

## 定义Scoring转换函数
def transform_Scoring(x):
    if pd.isnull(x):  # 检查是否为NaN
        return 0  # 如果是NaN，返回【0】
    else:
        return 201 - x  # 对非NaN值应用分数转换规则

## 应用函数到Rerank
Scoring = Rerankint.applymap(transform_Scoring).transpose()


# === Ordering + 标准化 ===

"""
def1
将数据进行标准化到自定义的范围内。
遍历Scoring每一列，将0到200之间的分数数值标准化至0到newmax之间，newmax为一个参数用于寻优
平均分返回Order

参数:
data (array): 原始数据
new_min (float): 缩放范围的最小值
new_max (float): 缩放范围的最大值

返回:
array: 标准化后的数据。
"""

def scale_order(Scoring, new_max):

    new_min=0
    normalized_columns = []
    StandardScore = pd.DataFrame()

    for column in Scoring.columns:
        # 每一列的最小值和最大值
        original_min = Scoring[column].min()
        original_max = Scoring[column].max()

        normalized_data = (Scoring[column] - original_min) / (original_max - original_min)
        scaled_data = normalized_data * (new_max - new_min) + new_min
        normalized_columns.append(scaled_data)

    StandardScore = pd.concat(normalized_columns, axis=1)
    StandardScore.columns = Scoring.columns
    Avgscore = StandardScore.mean()
    StandardScore.loc['Average'] = StandardScore.mean()
    
    order = Avgscore.sort_values(ascending=False).index
    StandardScore = StandardScore[order].round(2)

    return order, StandardScore

#new_max = 10
order = scale_order(Scoring, new_max)[0]
StandardScore = scale_order(Scoring, new_max)[1]
#order.tolist()


# === 数据重载入 ===


exp_data = pd.read_csv('C:/Users/14816/Desktop/Workdir/Done/BRCA.sampleMap_HiSeqV2', sep='\t')
gene_name = exp_data['sample']
exp_data = exp_data.drop(labels = 'sample',axis=1).transpose()
exp_data.columns = gene_name

ylable= pd.DataFrame(pd.DataFrame(exp_data.index)[0].str.split('-').str[-1])
exp_data['ylable'] = np.where(ylable[0].str.contains("0"), 1, 0)
#exp_data.to_csv('%s/BRCA.txt'%workdir,sep='\t',index='sample')

x = exp_data.iloc[:,0:20531]
x = x.drop(columns = 'ylable')
y = exp_data.iloc[:,-1].astype(int)


# === 独立测试集的封装 ===

x_data, x_final, y_data, y_final = train_test_split(x, y, test_size=0.2, random_state=15, shuffle=True, stratify=y)


# === Order的验证：逐个引入 ===

cv = StratifiedKFold(n_splits=10, random_state=15, shuffle=True)
for train_index, test_index in cv.split(x_data, y_data):
    x_train = x_data.iloc[train_index]
    y_train = y_data.iloc[train_index]
    x_test = x_data.iloc[test_index]
    y_test = y_data.iloc[test_index]

    AUC_list = []
    feature_number = []
    avg_AUC = []
    feature_list = order.tolist()

## === 不剔除：结果以折线图展示 ===

    def Step_in(x_train, feature_list, num):
        
        for i in range(1, num):
            current_features = feature_list[:i]
            subtrain = x_train[current_features].copy()

            svm_model = SVC(probability=True)
            scoring = {'AUC': 'roc_auc'}
            results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
                
            auc_scores = results['test_AUC']
            max_auc = max(auc_scores)
            average_auc = results['test_AUC'].mean()
            AUC_list.append(auc_scores)
            feature_number.append(i)
            avg_AUC.append(average_auc)

            plt.figure(figsize=(10, 6))
            plt.plot(feature_number, avg_AUC, marker='o', linewidth=1, markersize=4)
            plt.title('Features Step In Score')
            plt.xlabel('Number of Features')
            plt.ylabel('AUC Score')
            plt.grid(True)
            plt.show()

        return max_auc, average_auc

    '''
    jupynotebook测试code：

        for i in range(1, len(feature_list) + 1):
        #for i in range(1, 201):

            current_features = feature_list[:i]
            subtrain = x_train[current_features].copy()

            svm_model = SVC(probability=True)
            scoring = {'AUC': 'roc_auc'}
            results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
                
            auc_scores = results['test_AUC']
            max_auc = max(auc_scores)
            average_auc = results['test_AUC'].mean()
            AUC_list.append(auc_scores)
            feature_number.append(i)
            avg_AUC.append(average_auc)
    '''


## === 筛选+剔除 ===

    def Step_in_out(x_train, feature_list, num):
        
        current_AUC = 0
        selected_features = []

        for i in range(1, num):
            current_features = feature_list[:i]
            subtrain = x_train[current_features].copy()

            svm_model = SVC(probability=True)
            scoring = {'AUC': 'roc_auc'}
            results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
                
            auc_scores = results['test_AUC']
            max_auc = max(auc_scores)
            average_auc = results['test_AUC'].mean()

            AUC_list.append(auc_scores)
            feature_number.append(i)
            avg_AUC.append(average_auc)


            #筛选：若引入的特征没用，则将该特征剔除；反之则加上
            if current_AUC < average_auc:

                selected_features.append(feature_list[i])
                current_AUC = average_auc
                print(f"特征 {selected_features} New AUC: {current_AUC}")

            #else:
            #    print(f"Skip")

    '''
    筛选结果：
    特征 ['MMP11', 'COL10A1', 'FIGF', 'PPP1R12B', 'TMEM220', 'MAMDC2'] New AUC: 1.0

    jupynotebook测试code：
    
        AUC_list = []
        feature_list = order.tolist()
        feature_number = []
        avg_AUC = []
        current_AUC = 0
        selected_features = []
        
        for i in range(1, len(feature_list) + 1):

            current_features = feature_list[:i]
            subtrain = x_train[current_features].copy()

            svm_model = SVC(probability=True)
            scoring = {'AUC': 'roc_auc'}
            results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
                
            auc_scores = results['test_AUC']
            max_auc = max(auc_scores)
            average_auc = results['test_AUC'].mean()

            AUC_list.append(auc_scores)
            feature_number.append(i)
            avg_AUC.append(average_auc)

            #筛选：若引入的特征没用，则将该特征剔除；反之则加上
            if current_AUC < average_auc:

                selected_features.append(feature_list[i])
                current_AUC = average_auc
                print(f"特征 {selected_features} New AUC: {current_AUC}")

            #else:
                #print(f"Skip")
    '''


# ================================================
# 参比：从原始基因集中，随机抽取任意六个基因进行10×测试

rd = random.sample(range(1, 200), 6)
cv = StratifiedKFold(n_splits=10, random_state=15, shuffle=True)
for train_index, test_index in cv.split(x_data, y_data):
    x_train = x_data.iloc[train_index]
    y_train = y_data.iloc[train_index]
    x_test = x_data.iloc[test_index]
    y_test = y_data.iloc[test_index]
    
    subtrain = x_train[gene_name[rd]].copy()
    svm_model = SVC(probability=True)
    scoring = {'AUC': 'roc_auc'}
    results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
    auc_scores = results['test_AUC']
    max_auc = max(auc_scores)
    average_auc = results['test_AUC'].mean()
    
    print(f"auc_scores {auc_scores} Random average_auc: {average_auc}")

finalresults = cross_validate(svm_model, x_final, y_final, cv=cv, scoring=scoring)
final_scores = finalresults['test_AUC']
finalmax_auc = max(final_scores)
finalaverage_auc = finalresults['test_AUC'].mean()

# 选出的基因子集进行10×测试

pick_list = ['MMP11', 'COL10A1', 'FIGF', 'PPP1R12B', 'TMEM220', 'MAMDC2']

for train_index, test_index in cv.split(x_data, y_data):
    x_train = x_data.iloc[train_index]
    y_train = y_data.iloc[train_index]
    x_test = x_data.iloc[test_index]
    y_test = y_data.iloc[test_index]
    
    subtrain = x_train[pick_list].copy()
    svm_model = SVC(probability=True)
    scoring = {'AUC': 'roc_auc'}
    results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
    auc_scores = results['test_AUC']
    max_auc = max(auc_scores)
    average_auc = results['test_AUC'].mean()
    
    print(f"auc_scores {auc_scores} Random average_auc: {average_auc}")

finalresultsp = cross_validate(svm_model, x_final, y_final, cv=cv, scoring=scoring)
final_scoresp = finalresultsp['test_AUC']
finalmax_aucp = max(final_scoresp)
finalaverage_aucp = finalresultsp['test_AUC'].mean()


'''
模块化：
cv = StratifiedKFold(n_splits=10, random_state=15, shuffle=True)
for train_index, test_index in cv.split(x_data, y_data):
    x_train = x_data.iloc[train_index]
    y_train = y_data.iloc[train_index]
    x_test = x_data.iloc[test_index]
    y_test = y_data.iloc[test_index]

    def evaluate_AUC(x_train, order):
        
        subtrain = x_train[order].copy()
        svm_model = SVC(probability=True)
        scoring = {'AUC': 'roc_auc'}
        results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
        
        auc_scores = results['test_AUC']
        max_auc = max(auc_scores)
        average_auc = results['test_AUC'].mean()

        return auc_scores, max_auc, average_auc

'''






## 寻优

from sklearn.model_selection import GridSearchCV # 导入参数寻优包

#定义评估函数

def evaluate_performance(data, new_max):
    # 对数据进行标准化
    scaled_data = custom_scale(data, new_min=0, new_max=newmax)
    
    # 评估性能
    # 评估逻辑是？
    performance = some_performance_metric(scaled_data)
    
    return performance


#手动实现
#new_max 的预定义范围设置

new_max_range = np.arange(50, 500, 50) # 指定new_max的范围

def grid_search_newmax(data, new_max_range):

    best_performance = float('-inf')
    best_new_max = None

    for new_max in new_max_range:
        performance = evaluate_performance(data, new_max)
        
        if performance > best_performance:
            best_performance = performance
            best_new_max = new_max
    
    return best_new_max, best_performance

#GridSearchCV
def scale_new_max():

    new_max_range = np.power(2, np.arange(-1, 6, 1.0)) # 指定new_max的范围
    parameters = dict(new_max = new_max_range) # 将new_max组成字典，用于参数的grid遍历
    
    performance = evaluate_performance(data, new_max)
    grid = GridSearchCV(performance, param_grid=parameters, cv=numOfFolds) # 创建一个GridSearchCV实例
    grid.fit(X, y) # grid寻优
    print("The best parameters are %s with a score of %g" % (grid.best_params_, grid.best_score_))
    return grid


if __name__ == '__main__':
