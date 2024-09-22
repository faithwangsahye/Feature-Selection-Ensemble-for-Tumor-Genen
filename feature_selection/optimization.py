import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def optimize_feature_selection(importance_list, x_data, y_data, x_final, y_final, max_features=200, cv_folds=5):
    """
    通过优化特征排序参数，寻找最佳特征子集并最大化分类准确性。

    参数:
    - importance_list: 特征重要性列表
    - x_data: 训练集特征
    - y_data: 训练集标签
    - x_final: 测试集特征
    - y_final: 测试集标签
    - max_features: 最大特征数量 (默认为 200)
    - cv_folds: 交叉验证折数 (默认为 5)

    返回值:
    - best_min: 最优min值
    - best_acc: 最佳准确性
    - optimization_ACC: 每次优化后的准确性列表
    - optimization_feature: 每次优化后的特征列表
    """
    from feature_selection_ensemble import process_data, scale_order, Step_in_out_ACC

    optimization_ACC = []
    optimization_feature = []
    best_min = 0
    best_acc = 0

    for Optimization_min in range(1, max_features + 1):
        # 获取特征评分并进行归一化
        Scoring = process_data(importance_list)
        order, StandardScore = scale_order(Scoring, Optimization_min)
        optimization_feature_list = order.tolist()

        # 使用交叉验证逐步选择特征并计算准确性
        current_ACC, selected_features = Step_in_out_ACC(x_data, y_data, optimization_feature_list, max_features, cv_folds)

        # 使用SVM模型进行测试集上的准确性评估
        svm_model = SVC(probability=True)
        svm_model.fit(x_data[selected_features], y_data)
        optimization_predictions = svm_model.predict(x_final[selected_features])
        acc = accuracy_score(y_final, optimization_predictions)

        # 记录每次优化的结果
        optimization_ACC.append(acc)
        optimization_feature.append(list(selected_features))

        # 更新最佳参数和准确性
        if acc > best_acc:
            best_min = Optimization_min
            best_acc = acc

        # 显示当前进度
        print(f'Optimization step {Optimization_min}/{max_features} - Current Accuracy: {acc}', end='\r')

    print(f"\nBest min value: {best_min}, Best accuracy: {best_acc}")
    return best_min, best_acc, optimization_ACC, optimization_feature


def evaluate_algorithms(importance_list, x_data, y_data, x_final, y_final, max_features=200, cv_folds=5):
    """
    评估各个特征选择算法，并计算其对测试集的分类准确性。

    参数:
    - importance_list: 特征重要性列表
    - x_data: 训练集特征
    - y_data: 训练集标签
    - x_final: 测试集特征
    - y_final: 测试集标签
    - max_features: 最大特征数量 (默认为 200)
    - cv_folds: 交叉验证折数 (默认为 5)

    返回值:
    - algorithm_accuracies: 每个算法的分类准确性列表
    - algorithm_features: 每个算法选择的特征列表
    """
    from feature_selection_ensemble import process_data, scale_order, Step_in_out_ACC

    algorithm_list = importance_list.columns  # 获取所有算法的列表
    algorithm_accuracies = []
    algorithm_features = []

    # 对每个算法进行评估
    for algorithm in algorithm_list:
        print(f"Evaluating {algorithm}...")

        # 获取单个算法的特征重要性
        algorithm_importance = pd.DataFrame(importance_list[algorithm])

        # 处理特征评分并进行归一化
        Scoring = process_data(algorithm_importance)
        order, StandardScore = scale_order(Scoring, 1)
        feature_list = order.tolist()

        # 使用交叉验证逐步选择特征并计算准确性
        current_ACC, selected_features = Step_in_out_ACC(x_data, y_data, feature_list, max_features, cv_folds)

        # 使用SVM模型进行测试集上的准确性评估
        svm_model = SVC(probability=True)
        svm_model.fit(x_data[selected_features], y_data)
        algorithm_predictions = svm_model.predict(x_final[selected_features])
        algorithm_acc = accuracy_score(y_final, algorithm_predictions)

        # 记录每个算法的准确性和所选特征
        algorithm_accuracies.append(algorithm_acc)
        algorithm_features.append(list(selected_features))

        print(f"{algorithm} Accuracy: {algorithm_acc}")

    return algorithm_accuracies, algorithm_features


def ensemble_evaluation(importance_list, x_data, y_data, x_final, y_final, new_min, max_features=200, cv_folds=5):
    """
    使用集成方法评估特征选择的效果，并计算其在测试集上的分类准确性。

    参数:
    - importance_list: 特征重要性列表
    - x_data: 训练集特征
    - y_data: 训练集标签
    - x_final: 测试集特征
    - y_final: 测试集标签
    - new_min: 最佳归一化参数
    - max_features: 最大特征数量 (默认为 200)
    - cv_folds: 交叉验证折数 (默认为 5)

    返回值:
    - integrated_acc: 集成方法的分类准确性
    - selected_features: 最终选择的特征列表
    """
    from feature_selection import process_data, scale_order, Step_in_out_ACC

    # 获取集成方法下的特征评分并归一化
    Scoring = process_data(importance_list)
    order, StandardScore = scale_order(Scoring, new_min)
    feature_list = order.tolist()

    # 使用交叉验证逐步选择特征并计算准确性
    current_ACC, selected_features = Step_in_out_ACC(x_data, y_data, feature_list, max_features, cv_folds)

    # 使用SVM模型进行测试集上的准确性评估
    svm_model = SVC(probability=True)
    svm_model.fit(x_data[selected_features], y_data)
    integrated_predictions = svm_model.predict(x_final[selected_features])
    integrated_acc = accuracy_score(y_final, integrated_predictions)

    print(f"Ensemble Accuracy: {integrated_acc}")
    return integrated_acc, selected_features
