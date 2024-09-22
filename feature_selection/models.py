import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from skfeature.function.similarity_based import reliefF, fisher_score, SPEC, lap_score
from skfeature.function.statistical_based import f_score
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W
from arfs.feature_selection import Leshy
from lightgbm import LGBMRegressor


def xgb_model(x_data, y_data, importance_list, gene_name):
    """
    使用XGBoost模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    importance_list_xgb = []
    model = xgb.XGBClassifier()
    model.fit(x_data, y_data)

    combine = zip(x_data.columns, model.feature_importances_)
    importance = {k: v for k, v in combine}
    importance_list_xgb.append(importance)
    
    df_importance = pd.DataFrame(importance_list_xgb).fillna(0)
    df_result = df_importance.mean(axis=0, numeric_only=True).sort_values(ascending=False)
    xgb_rank = pd.DataFrame(df_result.index, columns=['XGBoost'])
    
    importance_list = pd.merge(importance_list, xgb_rank, left_index=True, right_index=True, how='outer')
    return importance_list


def lgb_model(x_data, y_data, importance_list, gene_name):
    """
    使用LightGBM模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    importance_list_lgb = []
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(x_data, y_data)
    
    importance = lgb_model.feature_importances_
    importance_list_lgb.append(importance)
    
    df_importance = pd.DataFrame(importance_list_lgb).fillna(0)
    df_importance = df_importance.rename(columns=gene_name)
    df_result = df_importance.mean(axis=0, numeric_only=True).sort_values(ascending=False)
    #df_result = pd.Series(df_importance.mean(axis=0, numeric_only=True), index=gene_name).sort_values(ascending=False)
    lgb_rank = pd.DataFrame(df_result.index, columns=['LightGBM'])
    
    importance_list = pd.merge(importance_list, lgb_rank, left_index=True, right_index=True, how='outer')
    return importance_list


def ctb_model(x_data, y_data, importance_list, gene_name):
    """
    使用CatBoost模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    importance_list_ctb = []

    ctb_model = CatBoostClassifier(iterations=10, silent=True)
    ctb_model.fit(x_data, y_data)
    
    importance = ctb_model.feature_importances_
    importance_list_ctb.append(importance)
    
    df_importance = pd.DataFrame(importance_list_ctb).fillna(0)
    df_importance = df_importance.rename(columns=gene_name)
    df_result = df_importance.mean(axis=0, numeric_only=True).sort_values(ascending=False)
    #df_result = pd.Series(df_importance.mean(axis=0), index=gene_name).sort_values(ascending=False)
    ctb_rank = pd.DataFrame(df_result.index, columns=['CatBoost'])
    
    importance_list = pd.merge(importance_list, ctb_rank, left_index=True, right_index=True, how='outer')
    return importance_list


def DTree_model(x_data, y_data, importance_list, gene_name):
    """
    使用决策树模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    clf = DecisionTreeClassifier()
    clf.fit(x_data, y_data)
    
    importance_scores = clf.feature_importances_
    feature_names = x_data.columns
    sorted_indices = importance_scores.argsort()[::-1]
    sorted_feature_names = feature_names[sorted_indices]
    sorted_feature_names = sorted_feature_names.tolist()
    
    df_L = pd.DataFrame(sorted_feature_names, columns=['DTree'])
    importance_list = pd.merge(importance_list, df_L, left_index=True, right_index=True, how='outer')
    return importance_list


def LRegression_model(x_data, y_data, importance_list, gene_name):
    """
    使用线性回归模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    model = LinearRegression()
    model.fit(x_data, y_data)
    
    coefficients = model.coef_
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    LR_names = x_data.columns
    LR_list = [LR_names[idx] for idx in sorted_indices]
    
    df_LR = pd.DataFrame(LR_list, columns=['LinearRegression'])
    importance_list = pd.merge(importance_list, df_LR, left_index=True, right_index=True, how='outer')
    return importance_list


def RFClassifier_model(x_data, y_data, importance_list, gene_name):
    """
    使用随机森林模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    model = RandomForestClassifier()
    model.fit(x_data, y_data)
    
    feature_importance = model.feature_importances_
    sorted_feature_importance = sorted(zip(x_data.columns, feature_importance), key=lambda x: x[1], reverse=True)
    
    df_rf = pd.DataFrame([x[0] for x in sorted_feature_importance], columns=['RFClassifier'])
    importance_list = pd.merge(importance_list, df_rf, left_index=True, right_index=True, how='outer')
    return importance_list


def adaboost_model(x_data, y_data, importance_list, gene_name):
    """
    使用AdaBoost模型进行训练，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost.fit(x_data, y_data)
    
    feature_importances = adaboost.feature_importances_
    ada_list = pd.DataFrame({'Feature': gene_name, 'Importance': feature_importances})
    ada_list.sort_values(by='Importance', ascending=False, inplace=True)
    
    df_abc = pd.DataFrame(ada_list['Feature'].tolist(), columns=['adaboost'])
    importance_list = pd.merge(importance_list, df_abc, left_index=True, right_index=True, how='outer')
    return importance_list


def reliefF_model(x_data, y_data, importance_list, gene_name):
    """
    使用ReliefF算法进行特征选择，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    x_data_values = x_data.values
    score = reliefF.reliefF(x_data_values, y_data)
    idx = reliefF.feature_ranking(score)
    
    df_rF = pd.DataFrame([gene_name[i] for i in idx], columns=['reliefF'])
    importance_list = pd.merge(importance_list, df_rF, left_index=True, right_index=True, how='outer')
    return importance_list


def Leshy_model(x_data, y_data, importance_list, gene_name):
    """
    使用Leshy算法进行特征选择并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    model = LGBMRegressor()
    feat_selector = Leshy(model, n_estimators=20, verbose=1, max_iter=10, random_state=111, importance="native")
    
    feat_selector.fit(x_data, y_data)
    ranks = feat_selector.ranking_
    arf_list = pd.DataFrame({'Feature': gene_name, 'Importance': ranks})
    arf_list.sort_values(by='Importance', ascending=False, inplace=True)  
    arfsfs_list=list(arf_list.iloc[:,0])
    
    name=['Leshy']
    name2=range(0,len(arfsfs_list))
    df_arfsfs=pd.DataFrame(columns=name,index=name2,data=arfsfs_list)
    importance_list = pd.merge(importance_list, df_arfsfs, left_index=True, right_index=True, how='outer')
    return importance_list


def MCFS_model(x_data, importance_list, gene_name):
    """
    使用MCFS算法进行特征选择，并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    W = construct_W.construct_W(x_data.values, metric="euclidean", neighborMode="knn", weightMode="heatKernel", k=5, t=1)
    score = MCFS.mcfs(x_data.values, n_selected_features=100, W=W, n_clusters=10)
    
    idx = MCFS.feature_ranking(score)
    df_MCFS = pd.DataFrame([gene_name[i] for i in idx], columns=['MCFS'])
    importance_list = pd.merge(importance_list, df_MCFS, left_index=True, right_index=True, how='outer')
    return importance_list


def fs_model(x_data,y_data, importance_list, gene_name):
    """
    使用Fisher score算法进行特征选择并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    score = fisher_score.fisher_score(x_data,y_data)
    idx = fisher_score.feature_ranking(score)
    df_fs = pd.DataFrame([gene_name[i] for i in idx], columns=['fisher_score'])
    importance_list = pd.merge(importance_list, df_fs, left_index=True, right_index=True, how='outer')
    
    return importance_list


def SPEC_model(x_data, importance_list, gene_name):
    """
    使用SPEC算法进行特征选择并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    x_data_1=x_data.values
    score = SPEC.spec(x_data_1)
    idx = SPEC.feature_ranking(score)
    df_SPEC = pd.DataFrame([gene_name[i] for i in idx], columns=['SPEC'])
    importance_list = pd.merge(importance_list, df_SPEC, left_index=True, right_index=True, how='outer')
    
    return importance_list


def f_model(x_data,y_data, importance_list, gene_name):
    """
    使用F_score算法进行特征选择并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - y_data: 训练数据标签
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    x_data_1=x_data.values
    score = f_score.f_score(x_data_1,y_data)
    idx = f_score.feature_ranking(score)
    df_f = pd.DataFrame([gene_name[i] for i in idx], columns=['f_score'])
    importance_list = pd.merge(importance_list, df_f, left_index=True, right_index=True, how='outer')
    
    return importance_list


def lap_model(x_data, importance_list, gene_name):
    """
    使用Lap_score算法进行特征选择并计算特征重要性。
    
    参数:
    - x_data: 训练数据特征
    - importance_list: 特征重要性列表
    - gene_name: 基因名称列表
    
    返回值:
    - 更新后的importance_list
    """
    x_data_1=x_data.values
    W = construct_W.construct_W(x_data_1, metric="euclidean", neighbor_mode="knn", weight_mode="heat_kernel", k=5, t=1)
    score = lap_score.lap_score(x_data_1,W=W)
    idx = lap_score.feature_ranking(score)
    df_lap = pd.DataFrame([gene_name[i] for i in idx], columns=['lap_score'])
    importance_list = pd.merge(importance_list, df_lap, left_index=True, right_index=True, how='outer')
    
    return importance_list