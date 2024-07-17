import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def data_import(file_path, label_col='sample', target_col='ylable', n_features=30557):
    exp_data = pd.read_csv(file_path, sep=',')
    exp_data.columns.values[0] = label_col
    gene_name = exp_data[label_col]
    exp_data = exp_data.drop(labels=label_col, axis=1).transpose()
    ylable = pd.DataFrame(exp_data.index.str.split('-').str[-1])
    exp_data.columns = gene_name
    del gene_name[n_features]
    
    x = exp_data.iloc[:, 0:n_features]
    y = exp_data.iloc[:, -1].astype(int)
    
    return x, y, gene_name

def Counting(importance_list, num=200):
    rankremain = importance_list[:num]
    gene_counts = rankremain.stack().value_counts()
    df_genecounts = pd.DataFrame(gene_counts.values, index=[gene_counts.index])
    df_genecounts.rename(columns={df_genecounts.columns[0]: 'Counts'}, inplace=True)
    return df_genecounts, rankremain

def RerankGenes(df_genecounts, rankremain):
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
    if pd.isna(x):
        return x
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x

def transform_Scoring(x):
    if pd.isnull(x):
        return 0
    else:
        return 200 - x

def process_data(importance_list):
    df_genecounts, rankremain = Counting(importance_list)
    Rerank = RerankGenes(df_genecounts, rankremain)
    Rerankint = Rerank.applymap(convert_to_numeric)
    Scoring = Rerankint.applymap(transform_Scoring).transpose()
    
    return Scoring

def scale_order(Scoring, new_min):
    normalized_columns = []
    StandardScore = pd.DataFrame()
    for column in Scoring.columns:
        # 每一列的最小值和最大值
        original_min = 1
        original_max = Scoring[column].max()
        new_max = 200
        #【添加一步判断：Scoring[column]的数值如果为0则保留为0，如果不等于0则进行以下分数转换】
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


'''
# === Order的验证：逐个引入 ===
基于train和final_rank(即Order)进行基因逐个引入
分为剔除与不剔除两种
【输入】
x_train：训练集特征, 
y_train：训练集标签, 
feature_list：特征列表, 
numb：纳入考虑的特征数量, 
cv_n：交叉验证折数
【输出】
average_acc：最后保留的特征子集的ACC,
savedfeat：最后保留的特征子集[这是一个列表]

'''

## === 不剔除 ===

def Step_in_AUC(x_train, y_train, feature_list, numb, cv_n):
    cv = StratifiedKFold(n_splits=cv_n, random_state=42, shuffle=True)
    avg_AUC = []
    for i in range(numb): #####range(numb)
        current_features = feature_list[:i+1]
        subtrain = x_train[current_features]

        svm_model = SVC(probability=True)
        scoring = {'AUC': 'roc_auc'}
        results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
            
        #auc_scores = results['test_AUC']
        #max_auc = max(auc_scores)
        average_auc = results['test_AUC'].mean()
        #AUC_list.append(auc_scores)
        #feature_number.append(i)
        avg_AUC.append(average_auc)
    
    savedfeat_num = avg_AUC.index(max(avg_AUC)) +1
    savedfeat = feature_list[:savedfeat_num]
    average_auc = avg_AUC[savedfeat_num -1]
    
    return average_auc,savedfeat

def Step_in_ACC(x_train, y_train, feature_list, numb, cv_n):
    cv = StratifiedKFold(n_splits=cv_n, random_state=42, shuffle=True)
    avg_ACC = []
    for i in range(numb): 
        current_features = feature_list[:i+1]
        subtrain = x_train[current_features]
        svm_model = SVC(probability=True)
        scoring = {'ACC': 'accuracy'}
        results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
        
        average_acc = results['test_ACC'].mean()
        avg_ACC.append(average_acc)
        
    savedfeat_num = avg_ACC.index(max(avg_ACC)) + 1
    savedfeat = feature_list[:savedfeat_num]
    average_acc = avg_ACC[savedfeat_num -1]
    
    return average_acc,savedfeat


## === 筛选+剔除 ===

def Step_in_out_AUC(x_train, y_train, feature_list, numb, cv_n):
    cv = StratifiedKFold(n_splits=cv_n, random_state=42, shuffle=True)
    current_AUC = [0]
    selected_features = []

    for i in range(numb):
        current_features = feature_list[i]
        subtrain = x_train[selected_features + [current_features]]
        svm_model = SVC(probability=True)
        scoring = {'AUC': 'roc_auc'}
        results = cross_validate(svm_model, subtrain, y_train, cv=cv, scoring=scoring)
            
        average_auc = results['test_AUC'].mean()
        #筛选：若引入的特征没用，则将该特征剔除；反之则加上
        if current_AUC[-1] < average_auc:
            selected_features.append(current_features)
            current_AUC.append(average_auc)
            print(f"特征 {selected_features} New AUC: {current_AUC}")
            
    return current_AUC, selected_features

def Step_in_out_ACC(x_train, y_train, feature_list, numb, cv_n):
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
            print(f"特征 {current_feature}. New ACC: {average_acc}")
            
    return current_ACC, selected_features



