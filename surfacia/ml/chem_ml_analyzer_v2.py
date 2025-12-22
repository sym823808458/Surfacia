"""
化学机器学习特征选择与分析工具包 - 完整版
适用于命令行环境
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import gc
from pathlib import Path
import shap
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端

# 设置默认随机种子
np.random.seed(42)

# ================== 基础分析器类 ==================
class BaseChemMLAnalyzer:
    """
    基础化学机器学习分析器 - 包含所有共享功能
    """
    
    def __init__(self, 
                 data_file, 
                 test_sample_names=None, 
                 nan_handling='drop_columns',
                 output_dir=None):
        """
        初始化基础分析器
        """
        self.data_file = data_file
        self.test_sample_names = test_sample_names or []
        self.nan_handling = nan_handling
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.output_dir = output_dir or str(Path(data_file).parent)
        
        # 设置随机种子
        np.random.seed(42)
        
        # XGBoost默认参数
        self.xgb_params = {
            'n_estimators': 350,
            'learning_rate': 0.03,
            'max_depth': 8,
            'verbosity': 0,
            'booster': 'gbtree',
            'reg_alpha': np.exp(-3),
            'reg_lambda': np.exp(-3),
            'gamma': np.exp(-5),
            'subsample': 0.5,
            'objective': 'reg:squarederror',
            'random_state': 42  # 添加随机种子
        }
        
        # 数据将在需要时加载
        self.data_loaded = False

    def _ensure_data_loaded(self):
        """确保数据已加载"""
        if not self.data_loaded:
            self._load_and_prepare_data()
            self.data_loaded = True

    def _load_and_prepare_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")

        # 读取原始CSV
        original_df = pd.read_csv(self.data_file)
        print(f"Original data shape: {original_df.shape}")

        # 处理NaN
        if self.nan_handling == 'drop_rows':
            original_df = original_df.dropna()
            print(f"After dropping NaN rows: {original_df.shape}")
        elif self.nan_handling == 'drop_columns':
            original_df = original_df.dropna(axis=1)
            print(f"After dropping NaN columns: {original_df.shape}")

        # 重置索引
        original_df = original_df.reset_index(drop=True)

        # 生成特征矩阵文件
        feature_files = self._generate_feature_matrix(original_df)

        # 加载数据
        self.X = np.loadtxt(feature_files['features'], delimiter=',')
        self.y = np.loadtxt(feature_files['values'])
        self.feature_names = np.loadtxt(feature_files['titles'], dtype=str, delimiter=',', comments='!')

        # 加载SMILES和原始数据
        self.smiles_data = np.loadtxt(feature_files['smiles'], dtype=str, delimiter=',', comments='!')
        self.sample_names = original_df['Sample Name'].values

        # 根据样本名称确定测试集
        test_mask = np.zeros(len(self.y), dtype=bool)
        test_indices_found = []
        test_names_found = []
        test_names_not_found = []

        if self.test_sample_names:
            for test_name in self.test_sample_names:
                matching_indices = np.where(self.sample_names == str(test_name))[0]
                if len(matching_indices) == 0:
                    try:
                        matching_indices = np.where(self.sample_names.astype(float) == float(test_name))[0]
                    except:
                        pass

                if len(matching_indices) > 0:
                    idx = matching_indices[0]
                    test_mask[idx] = True
                    test_indices_found.append(idx + 1)
                    test_names_found.append(str(test_name))
                else:
                    test_names_not_found.append(str(test_name))

            if test_names_found:
                print(f"Found {len(test_names_found)} test samples:")
                for i, (name, idx) in enumerate(zip(test_names_found, test_indices_found)):
                    print(f"  {i+1}. Sample '{name}' at position {idx}")

            if test_names_not_found:
                print(f"Warning: {len(test_names_not_found)} test samples not found:")
                for name in test_names_not_found:
                    print(f"  - Sample '{name}'")
        
        # 分割数据
        self.X_train = self.X[~test_mask]
        self.y_train = self.y[~test_mask]
        self.X_realtest = self.X[test_mask]
        self.y_realtest = self.y[test_mask]
        self.smiles_train = self.smiles_data[~test_mask]
        self.smiles_realtest = self.smiles_data[test_mask]
        self.sample_names_train = self.sample_names[~test_mask]
        self.sample_names_realtest = self.sample_names[test_mask]
        
        # 记录数据集信息
        self.dataset_info = {
            'original_samples': len(self.y),
            'original_features': len(self.feature_names),
            'train_samples': len(self.y_train),
            'test_samples': len(self.y_realtest),
            'test_sample_names_input': self.test_sample_names,
            'test_sample_names_found': test_names_found,
            'test_sample_names_not_found': test_names_not_found,
            'test_indices_1based': test_indices_found,
            'nan_handling_method': self.nan_handling,
            'train_test_split_method': 'Sample name specification' if self.test_sample_names else 'No test set specified'
        }
        
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Train set: {len(self.y_train)} samples")
        print(f"Test set: {len(self.y_realtest)} samples")

    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件 - 子类应该重写此方法"""
        raise NotImplementedError("Subclasses must implement _generate_feature_matrix")

    def _setup_output_directory(self, dir_prefix):
        """设置输出目录"""
        DIR_name = f'{dir_prefix}_{self.timestamp}'
        self.ML_DIR = Path(self.output_dir, DIR_name)
        os.makedirs(self.ML_DIR, exist_ok=True)
        return self.ML_DIR

    def _save_feature_files(self, merged_df):
        """保存特征文件"""
        S_N = merged_df.shape[0]
        F_N = merged_df.shape[1] - 3
        
        files = {}
        files['smiles'] = str(self.ML_DIR / f'Smiles_{S_N}.csv')
        files['values'] = str(self.ML_DIR / f'Values_True_{S_N}.csv')
        files['features'] = str(self.ML_DIR / f'Features_{S_N}_{F_N}.csv')
        files['titles'] = str(self.ML_DIR / f'Title_{F_N}.csv')
        
        merged_df[['smiles']].to_csv(files['smiles'], index=False, header=False)
        merged_df[['target']].to_csv(files['values'], index=False, header=False)
        merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1)\
                 .to_csv(files['features'], index=False, float_format='%.6f', header=False)
        
        with open(files['titles'], 'w') as f:
            for col in merged_df.columns[1:-2]:
                formatted_col = col.replace(' ', '_').replace('/', '_')
                f.write(formatted_col + '\n')
        
        return files

    def XGB_Fit(self, X, y, X_train, y_train, X_test, y_test, test_idx, paras):
        """单次XGBoost训练"""
        clf_new = XGBRegressor()
        for k, v in paras.items():
            clf_new.set_params(**{k: v})
        
        clf_new.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = clf_new.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        shap_values = shap.TreeExplainer(clf_new).shap_values(X_test, check_additivity=False)
        
        return ([mse, mae, r2, shap_values, test_idx], clf_new)

    def _single_fit_wrapper(self, X, y, train_idx, test_idx, paras, X_realtest=None):
        """单次训练的包装函数"""
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]
        
        results = self.XGB_Fit(X, y, X_train, y_train, X_test, y_test, test_idx, paras)
        
        return results, train_idx, test_idx

    def poolfit_optimized(self, TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, X, y, paras, X_realtest=None):
        """优化的并行训练函数，支持测试集SHAP值计算"""
        print(f"Starting optimized training: {EPOCH} epochs, {CORE_NUM} cores")
        
        point = round(X.shape[0] * TRAIN_TEST_SPLIT)
        
        # 初始化存储
        sample_shap_values = {}
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                sample_shap_values[(i, j)] = []
    
        # 生成所有的训练-测试划分（使用固定种子）
        all_splits = []
        np.random.seed(42)  # 确保可重现
        for epoch in range(EPOCH):
            permutation = np.random.permutation(y.shape[0])
            train_idx = permutation[:point]
            test_idx = permutation[point:]
            all_splits.append((train_idx, test_idx))
    
        # 分批处理
        batch_size = max(1, CORE_NUM)
        all_results = []
        
        for batch_start in range(0, EPOCH, batch_size):
            batch_end = min(batch_start + batch_size, EPOCH)
            batch_splits = all_splits[batch_start:batch_end]
            
            try:
                batch_results = Parallel(
                    n_jobs=CORE_NUM, 
                    backend='threading',
                    timeout=300,
                    verbose=0
                )(
                    delayed(self._single_fit_wrapper)(X, y, train_idx, test_idx, paras, X_realtest)
                    for train_idx, test_idx in batch_splits
                )
                
                all_results.extend(batch_results)
                gc.collect()
                
            except Exception as e:
                print(f"Batch processing failed: {e}")
                print("Falling back to single-threaded processing...")
                
                for train_idx, test_idx in batch_splits:
                    try:
                        result = self._single_fit_wrapper(X, y, train_idx, test_idx, paras, X_realtest)
                        all_results.append(result)
                    except Exception as e2:
                        print(f"Single fit also failed: {e2}")
                        continue
    
        if not all_results:
            raise RuntimeError("All training attempts failed")
    
        print(f"Successfully completed {len(all_results)} training rounds")
    
        # 处理结果
        mse_list = []
        mae_list = []
        r2_list = []
        full_m = np.zeros((len(all_results), X.shape[0]))
        y_realtest_pred_list = []
        realtest_shap_list = []  # 新增：存储测试集SHAP值
        split_l = []
        test_idx_m = []
        
        sample_counts = np.zeros((X.shape[0], X.shape[1]))
    
        for i, (results_tuple, train_idx, test_idx) in enumerate(all_results):
            temp = results_tuple[0]
            mse_list.append(temp[0])
            mae_list.append(temp[1])
            r2_list.append(temp[2])
            
            shap_values = temp[3]
            test_indices = temp[4]
            
            for j, test_idx_val in enumerate(test_indices):
                for k in range(X.shape[1]):
                    sample_shap_values[(test_idx_val, k)].append(shap_values[j, k])
                    sample_counts[test_idx_val, k] += 1
            
            clf_new = results_tuple[1]
            y_full_pred = clf_new.predict(X)
            full_m[i] = y_full_pred
            
            split_l.append(train_idx)
            test_idx_m.append(test_idx)
    
            # 计算真实测试集的预测和SHAP值
            if X_realtest is not None:
                y_realtest_pred = clf_new.predict(X_realtest)
                y_realtest_pred_list.append(y_realtest_pred)
                
                # 计算测试集SHAP值
                realtest_shap = shap.TreeExplainer(clf_new).shap_values(X_realtest, check_additivity=False)
                realtest_shap_list.append(realtest_shap)
    
        # 计算平均SHAP值和标准差（训练集）
        shap_m2 = np.zeros((X.shape[0], X.shape[1]))
        shap_std = np.zeros((X.shape[0], X.shape[1]))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if sample_counts[i, j] > 0:
                    values = sample_shap_values[(i, j)]
                    shap_m2[i, j] = np.mean(values) if values else 0
                    shap_std[i, j] = np.std(values) if len(values) > 1 else 0
    
        # 计算测试集的平均SHAP值和标准差
        realtest_shap_mean = None
        realtest_shap_std = None
        if realtest_shap_list:
            realtest_shap_mean = np.mean(realtest_shap_list, axis=0)
            realtest_shap_std = np.std(realtest_shap_list, axis=0)
    
        # 修改：根据最终散点图数据重新计算性能指标
        test_data_m = [[] for _ in range(len(y))]
        for i in range(test_idx_m.__len__()):
            for k in range(test_idx_m[i].__len__()):
                test_data_m[test_idx_m[i][k]].append(full_m[i, test_idx_m[i][k]])

        test_mean_l = [np.mean(item) if item else 0 for item in test_data_m]
        test_std_l = [np.std(item) if item else 0 for item in test_data_m]
        true_y = y.flatten().tolist()

        # 基于散点图数据重新计算性能指标
        mse1 = mean_squared_error(true_y, test_mean_l)
        mae1 = mean_absolute_error(true_y, test_mean_l)
        r21 = r2_score(true_y, test_mean_l)
        mse2 = np.std([mean_squared_error(true_y, [test_data_m[i][j] if test_data_m[i] else 0 
                                                  for i in range(len(true_y))]) 
                      for j in range(len(all_results)) if j < min([len(lst) for lst in test_data_m if lst])])
    
        y_realtest_pred_std = np.std(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
        y_realtest_pred_mean = np.mean(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
    
        print(f"Training completed: MSE={mse1:.4f}±{mse2:.4f}, MAE={mae1:.4f}, R²={r21:.4f}")
    
        return {
            'mse': mse1,
            'mae': mae1,
            'r2': r21,
            'mse_std': mse2,
            'full_predictions': full_m,
            'test_indices': test_idx_m,
            'shap_values': shap_m2,
            'shap_std': shap_std,
            'y_realtest_pred': y_realtest_pred_mean,
            'y_realtest_pred_std': y_realtest_pred_std,
            'realtest_shap_mean': realtest_shap_mean,  # 新增
            'realtest_shap_std': realtest_shap_std,    # 新增
            'scatter_data': {'true_y': true_y, 'test_mean_l': test_mean_l, 'test_std_l': test_std_l}  # 新增散点图数据
        }

    def save_comprehensive_results(self, save_dir, prefix, feature_names, feature_indices, 
                                  X_selected, perf_dict):
        """保存综合结果，包含测试集SHAP值"""
        print(f"Saving comprehensive results...")
        
        # 保存训练集详细CSV
        train_data = []
        
        # 添加特征列
        for i, feat_name in enumerate(feature_names):
            train_data.append(X_selected[:, i])
        
        # 添加SHAP列
        shap_values = perf_dict['shap_values']
        shap_std = perf_dict['shap_std']
        
        for i, feat_name in enumerate(feature_names):
            train_data.append(shap_values[:, i])
        
        # 添加SHAP标准差列
        for i, feat_name in enumerate(feature_names):
            train_data.append(shap_std[:, i])
        
        # 添加其他信息
        train_data.append(self.smiles_train)
        train_data.append(self.y_train)
        train_data.append(self.sample_names_train)
        
        # 转置以便按行保存
        train_data = np.column_stack(train_data)
        
        # 创建列名
        feature_cols = [f"Feature_{feat}" for feat in feature_names]
        shap_cols = [f"SHAP_{feat}" for feat in feature_names]
        shap_std_cols = [f"SHAP_Std_{feat}" for feat in feature_names]
        other_cols = ["SMILES", "Target", "Sample_Name"]
        column_names = feature_cols + shap_cols + shap_std_cols + other_cols
        
        # 保存训练集详细CSV
        train_df = pd.DataFrame(train_data, columns=column_names)
        save_path = save_dir / f'Training_Set_Detailed_{prefix}_{self.timestamp}.csv'
        train_df.to_csv(save_path, index=False)
        
        # 保存测试集CSV（包含SHAP值）
        if len(self.X_realtest) > 0:
            X_realtest_selected = self.X_realtest[:, feature_indices]
            
            test_data = []
            
            # 添加特征列
            for i, feat_name in enumerate(feature_names):
                test_data.append(X_realtest_selected[:, i])
            
            # 添加测试集SHAP列（如果有）
            if perf_dict.get('realtest_shap_mean') is not None:
                for i, feat_name in enumerate(feature_names):
                    test_data.append(perf_dict['realtest_shap_mean'][:, i])
                
                # 添加测试集SHAP标准差列
                for i, feat_name in enumerate(feature_names):
                    test_data.append(perf_dict['realtest_shap_std'][:, i])
                
                shap_test_cols = [f"SHAP_{feat}" for feat in feature_names]
                shap_test_std_cols = [f"SHAP_Std_{feat}" for feat in feature_names]
            else:
                shap_test_cols = []
                shap_test_std_cols = []
            
            # 添加其他信息
            test_data.append(self.smiles_realtest)
            test_data.append(self.y_realtest)
            test_data.append(self.sample_names_realtest)
            
            # 添加预测值
            y_realtest_pred = perf_dict['y_realtest_pred']
            y_realtest_pred_std = perf_dict['y_realtest_pred_std']
            if y_realtest_pred is not None:
                test_data.append(y_realtest_pred)
                test_data.append(y_realtest_pred_std)
                prediction_cols = ["Realtest_Pred", "Realtest_Pred_Std"]
            else:
                prediction_cols = []
            
            test_data = np.column_stack(test_data)
            
            # 创建列名
            test_feature_cols = [f"Feature_{feat}" for feat in feature_names]
            test_other_cols = ["SMILES", "Target", "Sample_Name"]
            test_column_names = test_feature_cols + shap_test_cols + shap_test_std_cols + test_other_cols + prediction_cols
            
            test_df = pd.DataFrame(test_data, columns=test_column_names)
            save_path = save_dir / f'Test_Set_Detailed_{prefix}_{self.timestamp}.csv'
            test_df.to_csv(save_path, index=False)

    def generate_prediction_scatter(self, perf_dict, feature_names, title_prefix="", save_dir=None):
        """生成预测散点图并保存原始数据"""
        if save_dir is None:
            save_dir = self.ML_DIR

        n_features = len(feature_names)
        
        # 使用修改后的散点图数据
        scatter_data = perf_dict['scatter_data']
        true_y = scatter_data['true_y']
        test_mean_l = scatter_data['test_mean_l']
        test_std_l = scatter_data['test_std_l']

        # 绘制散点图
        plt.figure(figsize=(12, 10), dpi=300)
        sc = plt.scatter(true_y, test_mean_l, alpha=0.55, c=test_std_l, cmap='viridis', marker='o', s=60)
        left_limit = min(min(true_y) - 1, min(test_mean_l) - 1)
        right_limit = max(max(true_y) + 1, max(test_mean_l) + 1)
        plt.plot([left_limit, right_limit], [left_limit, right_limit],
                 color='#B22222', linestyle=':', linewidth=3)
        plt.plot([left_limit, right_limit], [left_limit + 1, right_limit + 1],
                 color='#FFA500', linestyle=':', linewidth=3)
        plt.plot([left_limit, right_limit], [left_limit - 1, right_limit - 1],
                 color='#FFA500', linestyle=':', linewidth=3)
        plt.legend(['Perfect Fit', '+1', '-1', 'Cross-Validation Predictions'],
                   loc='upper left', fontsize=20, shadow=True)
        plt.xlabel('True Values', fontsize=24)
        plt.ylabel('Mean Predicted Values', fontsize=24)
        plt.title(f'{title_prefix} Model: Predicted vs True Values ({n_features} Features)\n'
                  f'MSE: {perf_dict["mse"]:.4f}±{perf_dict["mse_std"]:.4f}  MAE: {perf_dict["mae"]:.4f}  R²: {perf_dict["r2"]:.4f}',
                  fontsize=26)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cb = plt.colorbar(sc)
        cb.set_label('Std of Predictions', fontsize=20)
        cb.ax.tick_params(labelsize=16)
        plt.grid(which='major', color='#D5D5D5', alpha=0.5)
        plt.tight_layout()

        # 保存图片
        save_name = save_dir / f'{title_prefix}_Prediction_Scatter_{n_features}feats_{self.timestamp}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存散点图原始数据（True vs Predicted）
        xgbscatterdata = np.column_stack((true_y, test_mean_l, test_std_l))
        data_save_name = save_dir / f'{title_prefix}_Prediction_Data_{n_features}_Features_MSE{round(perf_dict["mse"],3)}_MSEstd{round(perf_dict["mse_std"],3)}_MAE{round(perf_dict["mae"],3)}_R2{round(perf_dict["r2"],3)}.txt'
        np.savetxt(data_save_name, xgbscatterdata, fmt='%.6f', delimiter=',', 
                   header='True_Value,Mean_Prediction,Std_Prediction', comments='')

        print(f"  Scatter plot saved: {save_name.name}")
        print(f"  Raw data saved: {data_save_name.name}")

    def generate_shap_plots(self, feat_vals, shap_vals, shap_std_vals, feat_name, save_dir):
        """生成SHAP分析图，包含拟合曲线"""
        # 创建SHAP图目录
        shap_plots_dir = save_dir / 'SHAP_Plots'
        os.makedirs(shap_plots_dir, exist_ok=True)

        # 创建原始数据目录
        shap_data_dir = save_dir / 'SHAP_Raw_Data'
        os.makedirs(shap_data_dir, exist_ok=True)

        # 保存原始数据
        raw_data = np.column_stack((feat_vals, shap_vals, shap_std_vals))
        data_save_path = shap_data_dir / f'{feat_name}_raw_data_{self.timestamp}.csv'
        np.savetxt(data_save_path, raw_data, fmt='%.6f', delimiter=',',
                  header='Feature_Value,SHAP_Value,SHAP_Std', comments='')

        # 生成SHAP散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(feat_vals, shap_vals, alpha=0.6, s=40, c='steelblue')
        plt.errorbar(feat_vals, shap_vals, yerr=shap_std_vals, 
                    fmt='none', ecolor='lightgray', alpha=0.3, capsize=0)

        # 尝试拟合
        try:
            fits, best_fit_type, best_params, best_r2 = self.fit_various_functions(feat_vals, shap_vals)

            if best_r2 > 0.1:  # 只有R²大于0.1时才画拟合线
                x_plot = np.linspace(min(feat_vals), max(feat_vals), 100)
                y_plot = self._eval_fit_func(x_plot, best_fit_type, best_params)

                # 获取拟合公式
                if best_fit_type in fits:
                    formula = fits[best_fit_type][2]
                else:
                    formula = f'{best_fit_type} fit'

                plt.plot(x_plot, y_plot, 'r-', linewidth=2, alpha=0.7,
                        label=f'{formula}\n(R²={best_r2:.3f})')
                plt.legend(fontsize=12, loc='best')

                # 保存拟合信息
                fit_info_path = shap_data_dir / f'{feat_name}_fit_info_{self.timestamp}.txt'
                with open(fit_info_path, 'w') as f:
                    f.write(f'Feature: {feat_name}\n')
                    f.write(f'Best fit type: {best_fit_type}\n')
                    f.write(f'R²: {best_r2:.4f}\n')
                    f.write(f'Formula: {formula}\n')
                    f.write('\nAll fits tested:\n')
                    for fit_type, (params, r2, formula) in fits.items():
                        f.write(f'  {fit_type}: R²={r2:.4f}, {formula}\n')
        except Exception as e:
            print(f"    Warning: Fitting failed for {feat_name}: {str(e)}")

        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel(f'{feat_name} Value', fontsize=16)
        plt.ylabel('SHAP Value', fontsize=16)
        plt.title(f'SHAP Analysis: {feat_name}', fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = shap_plots_dir / f'SHAP_{feat_name}_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def fit_various_functions(self, x, y, min_points=10, r2_threshold=0.01):
        """对SHAP散点图进行多种函数拟合并进行质量评估"""
        def calc_r2(y_true, y_pred):
            """计算R²值，处理边界情况"""
            if len(y_true) < 2:
                return 0
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot < 1e-10:  # 几乎为常数
                return 0
            r2 = 1 - (ss_res / ss_tot)
            return max(0, min(1, r2))  # 确保R²在[0,1]范围内

        def format_number(num):
            """格式化数字，处理科学计数法"""
            if abs(num) < 1e-3 or abs(num) > 1e3:
                return f"{num:.2e}"
            return f"{num:.3f}"

        def is_categorical(x, max_unique=5):
            """判断是否为分类变量"""
            unique_vals = np.unique(x)
            return len(unique_vals) <= max_unique

        def fit_categorical(x, y):
            """对分类变量进行拟合"""
            unique_vals = np.unique(x)
            means = {}
            for val in unique_vals:
                mask = x == val
                if np.sum(mask) > 0:
                    means[val] = np.mean(y[mask])

            # 创建预测值数组
            y_pred = np.zeros_like(y)
            for val, mean_val in means.items():
                y_pred[x == val] = mean_val

            # 计算R²
            r2 = calc_r2(y, y_pred)

            # 创建公式字符串
            formula_parts = [f"y = {format_number(v)} (x = {format_number(k)})" 
                            for k, v in means.items()]
            formula = ", ".join(formula_parts)

            return means, r2, formula

        # 初始化
        fits = {}
        best_r2 = -np.inf
        best_fit_type = None
        best_params = None

        # 检查数据量
        if len(x) < min_points:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        # 检查是否为分类变量
        if is_categorical(x):
            params, r2, formula = fit_categorical(x, y)
            if r2 > r2_threshold:
                fits['categorical'] = (params, r2, formula)
                return fits, 'categorical', params, r2

        # 对于连续变量，进行标准化
        x_scale = np.std(x)
        y_scale = np.std(y)
        if x_scale < 1e-10 or y_scale < 1e-10:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        try:
            # 1. 多项式拟合
            for deg in range(1, 4):
                try:
                    coeffs = np.polyfit(x, y, deg)
                    y_pred = np.polyval(coeffs, x)
                    r2 = calc_r2(y, y_pred)

                    if r2 > r2_threshold:
                        terms = []
                        for i, c in enumerate(coeffs[::-1]):
                            if abs(c) < 1e-10:
                                continue
                            if i == 0:
                                terms.append(format_number(c))
                            elif i == 1:
                                terms.append(f"{format_number(c)}x")
                            else:
                                terms.append(f"{format_number(c)}x^{i}")
                        formula = "y = " + " + ".join(reversed(terms))

                        fits[f'poly_{deg}'] = (coeffs, r2, formula)

                        if r2 > best_r2:
                            best_r2 = r2
                            best_fit_type = f'poly_{deg}'
                            best_params = coeffs
                except:
                    continue

            # 2. 指数函数拟合
            def exp_func(x_, a, b, c):
                return a * np.exp(b * x_) + c

            try:
                p0 = [(np.max(y) - np.min(y)), 1.0 / x_scale, np.min(y)]
                popt, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=5000)
                y_pred = exp_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}*exp({format_number(popt[1])}x) + {format_number(popt[2])}"
                    fits['exp'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'exp'
                        best_params = popt
            except:
                pass

            # 3. 对数函数拟合
            def log_func(x_, a, b, c):
                return a * np.log(x_ + b) + c

            try:
                min_x_shift = max(0, -np.min(x)) + 1e-3
                p0 = [(np.max(y) - np.min(y)) / np.log(10), min_x_shift, np.min(y)]
                popt, _ = curve_fit(log_func, x, y, p0=p0, maxfev=5000)
                y_pred = log_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}*ln(x + {format_number(popt[1])}) + {format_number(popt[2])}"
                    fits['log'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'log'
                        best_params = popt
            except:
                pass

            # 4. Sigmoid函数拟合
            def sigmoid_func(x_, a, b, c, d):
                return a / (1 + np.exp(-b * (x_ - c))) + d

            try:
                p0 = [max(y) - min(y), 4 / x_scale, np.median(x), np.min(y)]
                popt, _ = curve_fit(sigmoid_func, x, y, p0=p0, maxfev=8000)
                y_pred = sigmoid_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}/(1 + e^(-{format_number(popt[1])}*(x - {format_number(popt[2])}))) + {format_number(popt[3])}"
                    fits['sigmoid'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'sigmoid'
                        best_params = popt
            except:
                pass

            # 5. 分段线性函数拟合
            percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            best_threshold_r2 = -np.inf
            best_threshold_params = None

            for p in percentiles:
                threshold = np.percentile(x, p)
                mask = x >= threshold

                if np.sum(mask) < min_points or np.sum(~mask) < min_points:
                    continue

                try:
                    left_value = np.mean(y[~mask]) if np.any(~mask) else 0
                    if np.sum(mask) >= 2:
                        right_coeffs = np.polyfit(x[mask], y[mask], 1)
                        y_pred = np.full_like(y, left_value)
                        y_pred[mask] = np.polyval(right_coeffs, x[mask])
                        r2 = calc_r2(y, y_pred)

                        if r2 > best_threshold_r2:
                            best_threshold_r2 = r2
                            best_threshold_params = (threshold, left_value, right_coeffs)
                except:
                    continue

            if best_threshold_r2 > r2_threshold:
                threshold, left_value, right_coeffs = best_threshold_params
                formula = f"y = {format_number(left_value)} (x < {format_number(threshold)}), {format_number(right_coeffs[0])}x + {format_number(right_coeffs[1])} (x >= {format_number(threshold)})"
                fits['piecewise'] = (best_threshold_params, best_threshold_r2, formula)

                if best_threshold_r2 > best_r2:
                    best_r2 = best_threshold_r2
                    best_fit_type = 'piecewise'
                    best_params = best_threshold_params

        except Exception as e:
            print(f"Warning: Fitting error occurred: {str(e)}")
            return {'error': ([], 0, 'fitting failed')}, 'error', [], 0

        if best_r2 < r2_threshold:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        return fits, best_fit_type, best_params, best_r2
    
    def _eval_fit_func(self, x, fit_type, params):
        """评估拟合函数，处理所有拟合类型"""
        if fit_type.startswith('poly_'):
            return np.polyval(params, x)
        elif fit_type == 'exp':
            a, b, c = params
            return a * np.exp(b * x) + c
        elif fit_type == 'log':
            a, b, c = params
            return a * np.log(x + b) + c
        elif fit_type == 'sigmoid':
            a, b, c, d = params
            return a / (1 + np.exp(-b * (x - c))) + d
        elif fit_type == 'categorical':
            y_pred = np.zeros_like(x)
            for val, mean_val in params.items():
                y_pred[x == val] = mean_val
            return y_pred
        elif fit_type == 'piecewise':
            threshold, left_value, right_coeffs = params
            y_pred = np.full_like(x, left_value)
            mask = x >= threshold
            y_pred[mask] = np.polyval(right_coeffs, x[mask])
            return y_pred
        return np.zeros_like(x)

    def _plot_stepwise_results(self):
        """可视化逐步回归结果（字号增大70%）"""
        plt.figure(figsize=(15, 10))
        
        for i, result in enumerate(self.stepwise_results):
            plt.plot(range(1, len(result['mse_history']) + 1), 
                    result['mse_history'], 
                    marker='o', linewidth=3, markersize=8,
                    label=f"Run {result['run_id']} (Final MSE: {result['final_mse']:.3f})")
        
        plt.xlabel('Number of Features', fontsize=24)
        plt.ylabel('MSE', fontsize=24)
        plt.title('Stepwise Regression Results - MSE vs Number of Features', fontsize=28)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.ML_DIR / f'stepwise_regression_results_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始数据
        stepwise_data = []
        max_features = max(len(result['mse_history']) for result in self.stepwise_results)
        
        # 创建数据矩阵，不足的用NaN填充
        for result in self.stepwise_results:
            row = result['mse_history'] + [np.nan] * (max_features - len(result['mse_history']))
            stepwise_data.append(row)
        
        stepwise_data = np.array(stepwise_data)
        
        # 创建列名
        columns = [f'Features_{i+1}' for i in range(max_features)]
        
        # 保存为CSV
        stepwise_df = pd.DataFrame(stepwise_data, columns=columns)
        stepwise_df.index = [f'Run_{i+1}' for i in range(len(self.stepwise_results))]
        data_save_path = self.ML_DIR / f'stepwise_regression_data_{self.timestamp}.csv'
        stepwise_df.to_csv(data_save_path)
        
        print(f"Stepwise regression plot saved: {save_path}")
        print(f"Stepwise regression data saved: {data_save_path}")

    def _plot_baseline_analysis(self):
        """绘制基线分析图表，包括散点图"""
        # 1. 特征重要性图（只显示前15个特征，字号增大70%）
        plt.figure(figsize=(15, 10))
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.baseline_feature_importance
        }).sort_values('importance', ascending=True)

        # 只取前15个特征
        n_features = min(15, len(importance_df))
        importance_df_top = importance_df.tail(n_features)

        plt.barh(range(len(importance_df_top)), importance_df_top['importance'])
        plt.yticks(range(len(importance_df_top)), importance_df_top['feature'], fontsize=20)
        plt.xlabel('Feature Importance', fontsize=24)
        plt.title('XGBoost Feature Importance (Top 15 Features)', fontsize=28)
        plt.xticks(fontsize=20)
        plt.tight_layout()

        # 保存特征重要性图
        importance_save_path = self.ML_DIR / f'baseline_feature_importance_{self.timestamp}.png'
        plt.savefig(importance_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存特征重要性原始数据
        importance_data_path = self.ML_DIR / f'baseline_feature_importance_data_{self.timestamp}.csv'
        importance_df.to_csv(importance_data_path, index=False)

        # 2. 相关性热图
        plt.figure(figsize=(20, 16))
        sns.heatmap(self.correlation_matrix, 
                   xticklabels=self.correlation_matrix.columns,
                   yticklabels=self.correlation_matrix.columns,
                   cmap='RdBu',
                   center=0,
                   annot=False,
                   square=True)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存相关性矩阵图
        correlation_save_path = self.ML_DIR / f'correlation_matrix_{self.timestamp}.png'
        plt.savefig(correlation_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存相关性矩阵原始数据
        correlation_data_path = self.ML_DIR / f'correlation_matrix_data_{self.timestamp}.csv'
        self.correlation_matrix.to_csv(correlation_data_path)

        print(f"Feature importance plot saved: {importance_save_path}")
        print(f"Feature importance data saved: {importance_data_path}")
        print(f"Correlation matrix plot saved: {correlation_save_path}")
        print(f"Correlation matrix data saved: {correlation_data_path}")

    def generate_record_file(self, save_dir, analysis_type, results, mode="auto"):
        """
        生成详细的记录文件

        Args:
            save_dir: 保存目录
            analysis_type: 分析类型名称
            results: 结果字典
            mode: "auto", "manual", "workflow", "baseline"
        """
        record_path = save_dir / f'Record_{analysis_type}_{self.timestamp}.txt'

        with open(record_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write(f'CHEMICAL MACHINE LEARNING - {analysis_type.upper()} ANALYSIS RECORD\n')
            f.write('='*80 + '\n\n')

            # 基本信息
            f.write('BASIC INFORMATION:\n')
            f.write('-'*40 + '\n')
            f.write(f'Analysis start time: {self.start_time}\n')
            f.write(f'Analysis end time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Timestamp: {self.timestamp}\n')
            f.write(f'Input data file: {self.data_file}\n')
            f.write(f'Output directory: {save_dir}\n\n')

            # 数据集信息
            f.write('DATASET INFORMATION:\n')
            f.write('-'*40 + '\n')
            f.write(f'Original samples: {self.dataset_info["original_samples"]}\n')
            f.write(f'Original features: {self.dataset_info["original_features"]}\n')
            f.write(f'Training samples: {self.dataset_info["train_samples"]}\n')
            f.write(f'Test samples: {self.dataset_info["test_samples"]}\n')
            f.write(f'NaN handling method: {self.dataset_info["nan_handling_method"]}\n\n')

            # 测试集信息
            if self.test_sample_names:
                f.write('TEST SET SPECIFICATION:\n')
                f.write('-'*40 + '\n')
                f.write(f'Input test sample names: {self.dataset_info["test_sample_names_input"]}\n')
                f.write(f'Successfully found {len(self.dataset_info["test_sample_names_found"])} samples:\n')
                for i, (name, idx) in enumerate(zip(self.dataset_info["test_sample_names_found"], 
                                                   self.dataset_info["test_indices_1based"])):
                    f.write(f'  {i+1}. "{name}" at position {idx}\n')

                if self.dataset_info["test_sample_names_not_found"]:
                    f.write(f'Warning: {len(self.dataset_info["test_sample_names_not_found"])} samples not found:\n')
                    for name in self.dataset_info["test_sample_names_not_found"]:
                        f.write(f'  - "{name}"\n')
                f.write('\n')

            # 特征选择信息
            if 'selected_features' in results:
                f.write('FEATURE SELECTION:\n')
                f.write('-'*40 + '\n')
                f.write(f'Number of selected features: {len(results["selected_features"])}\n')
                f.write('Selected features:\n')
                for i, feat in enumerate(results["selected_features"][:20]):  # 最多显示20个
                    f.write(f'  {i+1}. {feat}\n')
                if len(results["selected_features"]) > 20:
                    f.write(f'  ... and {len(results["selected_features"]) - 20} more features\n')
                f.write('\n')

            # 模型参数
            f.write('MODEL PARAMETERS:\n')
            f.write('-'*40 + '\n')
            f.write('XGBoost parameters:\n')
            for key, value in self.xgb_params.items():
                f.write(f'  {key}: {value}\n')
            f.write('\n')

            # 性能结果
            f.write('PERFORMANCE RESULTS:\n')
            f.write('-'*40 + '\n')
            f.write(f'Cross-validation performance (calculated from final scatter plot):\n')
            f.write(f'  MSE: {results["mse"]:.6f} ± {results["mse_std"]:.6f}\n')
            f.write(f'  MAE: {results["mae"]:.6f}\n')
            f.write(f'  R²: {results["r2"]:.6f}\n\n')

            if 'test_performance' in results and results['test_performance']:
                f.write('Test set performance:\n')
                f.write(f'  Test MSE: {results["test_performance"]["mse"]:.6f}\n')
                f.write(f'  Test MAE: {results["test_performance"]["mae"]:.6f}\n')
                f.write(f'  Test R²: {results["test_performance"]["r2"]:.6f}\n\n')

                # 详细预测结果（如果测试集较小）
                if hasattr(self, 'y_realtest') and len(self.y_realtest) <= 20:
                    f.write('DETAILED TEST PREDICTIONS:\n')
                    f.write('-'*40 + '\n')
                    if 'y_realtest_pred' in results:
                        for i, (true_val, sample_name) in enumerate(zip(self.y_realtest, self.sample_names_realtest)):
                            f.write(f'  {i+1}. {sample_name}:\n')
                            f.write(f'     True value: {true_val:.6f}\n')
                            if results.get('y_realtest_pred') is not None:
                                pred_val = results['y_realtest_pred'][i]
                                pred_std = results['y_realtest_pred_std'][i] if results.get('y_realtest_pred_std') is not None else 0
                                f.write(f'     Predicted: {pred_val:.6f} ± {pred_std:.6f}\n')
                                f.write(f'     Error: {abs(true_val - pred_val):.6f}\n')
                            f.write('\n')

            # 输出文件信息
            f.write('OUTPUT FILES:\n')
            f.write('-'*40 + '\n')

            if mode == "workflow":
                f.write('1. Baseline analysis:\n')
                f.write('   - Baseline_Analysis/ folder\n')
                f.write('2. Stepwise regression:\n')
                f.write('   - Run_1/, Run_2/, Run_3/ folders\n')
                f.write('3. SHAP analysis:\n')
                f.write('   - SHAP_Plots/ and SHAP_Raw_Data/ in each Run folder\n')
                f.write('4. Final analysis:\n')
                f.write('   - Final_Manual_Analysis/ folder\n')
                f.write('5. Recommendations:\n')
                f.write(f'   - feature_recommendations_{self.timestamp}.csv\n')
                f.write(f'   - workflow_summary_{self.timestamp}.txt\n')
            else:
                n_features = len(results.get("selected_features", self.feature_names))
                f.write(f'1. Prediction scatter plot and data\n')
                f.write(f'2. Training set details CSV\n')
                f.write(f'3. Test set details CSV (with SHAP values)\n')
                f.write(f'4. SHAP plots in SHAP_Plots/ folder\n')
                f.write(f'5. SHAP raw data in SHAP_Raw_Data/ folder\n')
                f.write(f'6. This record file\n')

            f.write('\n')
            f.write('='*80 + '\n')
            f.write(f'{analysis_type.upper()} ANALYSIS COMPLETED SUCCESSFULLY\n')
            f.write('='*80 + '\n')

# ================== 手动特征分析器 ==================
class ManualFeatureAnalyzer(BaseChemMLAnalyzer):
    """
    手动特征分析器 - 分析用户指定的特征
    """
    
    def __init__(self, data_file, test_sample_names=None, nan_handling='drop_columns', 
                 output_dir=None, features=None):
        """
        初始化手动特征分析器
        
        Args:
            features: 要分析的特征名称列表，或 'Full' 表示所有特征
        """
        self.manual_features = features or []
        super().__init__(data_file, test_sample_names, nan_handling, output_dir)
        
    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件"""
        if self.manual_features == 'Full':
            self._setup_output_directory('Full_Feature_Analysis')
        else:
            self._setup_output_directory('Manual_Feature_Analysis')
        return self._save_feature_files(merged_df)
    
    def run(self, epoch=64, core_num=32, train_test_split=0.85, generate_fitting=True):
        """
        运行手动特征分析

        Returns:
            dict: 包含分析结果的字典
        """
        self._ensure_data_loaded()

        print("\n" + "="*60)
        print("MANUAL FEATURE ANALYSIS")
        print("="*60)

        # 确定要使用的特征
        if self.manual_features == 'Full':
            print("Using ALL features for analysis")
            selected_indices = list(range(len(self.feature_names)))
            selected_features = self.feature_names.tolist()

            # 在全特征模式下添加额外的基线分析（但不做逐步回归）
            self._run_full_feature_baseline_analysis(epoch, core_num, train_test_split)

        else:
            # 验证特征
            selected_indices = []
            selected_features = []

            print(f"Validating {len(self.manual_features)} specified features...")
            for feat_name in self.manual_features:
                feat_idx_array = np.where(self.feature_names == feat_name)[0]
                if len(feat_idx_array) == 0:
                    print(f"Warning: Feature '{feat_name}' not found!")
                    raise ValueError(f"Feature '{feat_name}' not found")
                selected_indices.append(feat_idx_array[0])
                selected_features.append(feat_name)

        print(f"\nAnalyzing {len(selected_features)} features")

        # 准备数据
        X_selected = self.X_train[:, selected_indices]
        X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None

        # 训练模型 - 这里使用传入的epoch参数
        print(f"\nTraining with {len(selected_features)} features using {epoch} epochs...")
        training_start = time.strftime("%Y-%m-%d %H:%M:%S")

        perf = self.poolfit_optimized(train_test_split, epoch, core_num, 
                                     X_selected, self.y_train, self.xgb_params, 
                                     X_realtest_selected)

        training_end = time.strftime("%Y-%m-%d %H:%M:%S")

        # 计算测试集性能
        test_performance = None
        if len(self.y_realtest) > 0 and perf['y_realtest_pred'] is not None:
            test_mae = mean_absolute_error(self.y_realtest, perf['y_realtest_pred'])
            test_mse = mean_squared_error(self.y_realtest, perf['y_realtest_pred'])
            test_r2 = r2_score(self.y_realtest, perf['y_realtest_pred'])
            test_performance = {
                'mae': test_mae,
                'mse': test_mse,
                'r2': test_r2
            }
            print(f"\nTest Set Performance:")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Test MSE: {test_mse:.4f}")
            print(f"  Test R²: {test_r2:.4f}")

        # 生成可视化
        self.generate_prediction_scatter(perf, selected_features, "Manual")

        # 生成SHAP分析（即使是Full模式也生成）
        if generate_fitting:
            print("Generating SHAP analysis...")
            # 限制SHAP图的数量，但Full模式也生成
            max_shap_plots = 30 if self.manual_features == 'Full' else 20
            for i, feat_name in enumerate(selected_features[:max_shap_plots]):
                self.generate_shap_plots(
                    X_selected[:, i],
                    perf['shap_values'][:, i],
                    perf['shap_std'][:, i],
                    feat_name,
                    self.ML_DIR
                )

        # 保存数据
        self.save_comprehensive_results(self.ML_DIR, f"Manual_{len(selected_features)}feats",
                                       selected_features, selected_indices, 
                                       X_selected, perf)

        # 组装结果
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'shap_values': perf['shap_values'],
            'shap_std': perf['shap_std'],
            'test_performance': test_performance,
            'training_start': training_start,
            'training_end': training_end,
            'output_dir': str(self.ML_DIR),
            'epoch_used': epoch  # 记录实际使用的epoch数
        }

        # 生成记录文件
        self.generate_record_file(self.ML_DIR, "Manual_Feature", results, "manual")

        print(f"\nResults: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, MAE={perf['mae']:.4f}, R²={perf['r2']:.4f}")
        print(f"Training used {epoch} epochs as specified")
        print(f"Results saved in: {self.ML_DIR}")

        return results

    def _run_full_feature_baseline_analysis(self, epoch, core_num, train_test_split):
        """在全特征模式下运行额外的基线分析"""
        print("\nRunning additional baseline analysis for full feature mode...")
        
        # 计算特征重要性
        clf_baseline = XGBRegressor(**self.xgb_params)
        clf_baseline.fit(self.X_train, self.y_train)
        self.baseline_feature_importance = clf_baseline.feature_importances_
        
        # 计算相关性矩阵
        feature_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.correlation_matrix = feature_df.corr()
        
        # 生成基线分析图表
        self._plot_baseline_analysis()
        
        # 保存全特征矩阵
        full_feature_path = self.ML_DIR / f'full_feature_matrix_{self.timestamp}.csv'
        full_feature_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        full_feature_df['Target'] = self.y_train
        full_feature_df['Sample_Name'] = self.sample_names_train
        full_feature_df['SMILES'] = self.smiles_train
        full_feature_df.to_csv(full_feature_path, index=False)
        
        print(f"Full feature matrix saved: {full_feature_path}")

# ================== 工作流分析器 ==================
class WorkflowAnalyzer(BaseChemMLAnalyzer):
    """
    完整工作流分析器 - 包含基线分析、逐步回归、SHAP分析和最终推荐
    """
    
    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件"""
        self._setup_output_directory('Workflow_Analysis')
        return self._save_feature_files(merged_df)
    
    def run(self, max_features=5, n_runs=3, epoch=64, core_num=32, train_test_split=0.85):
        """
        运行完整工作流
        """
        self._ensure_data_loaded()
        
        print("\n" + "="*60)
        print("COMPLETE WORKFLOW ANALYSIS")
        print("="*60)
        
        all_results = {}
        
        # Step 1: 基线分析（全特征）
        print("\n" + "="*50)
        print("STEP 1: BASELINE ANALYSIS WITH ALL FEATURES")
        print("="*50)
        baseline_results = self._run_baseline_analysis(epoch, core_num, train_test_split)
        all_results['baseline'] = baseline_results
        
        # 清理内存
        gc.collect()
        
        # Step 2: 多轮逐步回归
        print("\n" + "="*50)
        print(f"STEP 2: STEPWISE REGRESSION ({n_runs} runs)")
        print("="*50)
        stepwise_results = self._run_stepwise_regression(max_features, n_runs, epoch, core_num, train_test_split)
        all_results['stepwise'] = stepwise_results
        
        # Step 3: SHAP拟合分析（对每个Run）
        print("\n" + "="*50)
        print("STEP 3: SHAP FITTING ANALYSIS")
        print("="*50)
        shap_analysis_results = self._run_shap_analysis(stepwise_results, epoch, core_num, train_test_split)
        all_results['shap_analysis'] = shap_analysis_results
        
        # Step 4: 智能推荐
        print("\n" + "="*50)
        print("STEP 4: INTELLIGENT RECOMMENDATION")
        print("="*50)
        recommendations = self._intelligent_recommendation(stepwise_results, baseline_results)
        all_results['recommendations'] = recommendations
        
        # Step 5: 使用推荐特征进行最终分析
        print("\n" + "="*50)
        print("STEP 5: FINAL ANALYSIS WITH RECOMMENDED FEATURES")
        print("="*50)
        
        top_features = [rec['feature_name'] for rec in recommendations[:max_features]]
        final_results = self._run_final_analysis(top_features, epoch, core_num, train_test_split)
        all_results['final'] = final_results
            
        # Step 6: 生成Final step_plot（n轮逐步回归+推荐特征曲线）
        print("\n" + "="*50)
        print("STEP 6: GENERATING FINAL STEPWISE PLOT")
        print("="*50)
        self._plot_final_stepwise_results(stepwise_results, final_results, max_features)

        # 保存工作流摘要，传入所有必需的参数
        self._save_workflow_summary(all_results, epoch, core_num, train_test_split, n_runs, max_features)
        
        print("\n" + "="*60)
        print("WORKFLOW ANALYSIS COMPLETED")
        print("="*60)
        print(f"Baseline MSE (all features): {baseline_results['mse']:.4f}")
        print(f"Final MSE ({len(top_features)} features): {final_results['mse']:.4f}")
        print(f"Recommended features: {top_features}")
        print(f"All results saved in: {self.ML_DIR}")
        
        return all_results
        
    def _plot_final_stepwise_results(self, stepwise_results, final_results, max_features):
        """可视化最终逐步回归结果（n轮逐步回归+推荐特征曲线）"""
        plt.figure(figsize=(15, 10))

        # 绘制n轮逐步回归曲线
        for i, result in enumerate(stepwise_results):
            plt.plot(range(1, len(result['mse_history']) + 1), 
                    result['mse_history'], 
                    marker='o', linewidth=3, markersize=8,
                    label=f"Run {result['run_id']} (Final MSE: {result['final_mse']:.3f})")

        # 添加推荐特征的性能点
        recommended_mse = final_results['mse']
        plt.scatter([max_features], [recommended_mse], 
                   marker='*', s=200, c='red', zorder=5,
                   label=f"Recommended Features (MSE: {recommended_mse:.3f})")

        plt.xlabel('Number of Features', fontsize=24)
        plt.ylabel('MSE', fontsize=24)
        plt.title('Final Stepwise Regression Results - MSE vs Number of Features', fontsize=28)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图片
        save_path = self.ML_DIR / f'final_stepwise_regression_results_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存原始数据
        stepwise_data = []
        max_features_in_runs = max(len(result['mse_history']) for result in stepwise_results)

        # 创建数据矩阵，不足的用NaN填充
        for result in stepwise_results:
            row = result['mse_history'] + [np.nan] * (max_features_in_runs - len(result['mse_history']))
            stepwise_data.append(row)

        stepwise_data = np.array(stepwise_data)

        # 创建列名
        columns = [f'Features_{i+1}' for i in range(max_features_in_runs)]

        # 保存为CSV
        stepwise_df = pd.DataFrame(stepwise_data, columns=columns)
        stepwise_df.index = [f'Run_{i+1}' for i in range(len(stepwise_results))]

        # 添加推荐特征行
        recommended_row = [np.nan] * max_features_in_runs
        if max_features <= max_features_in_runs:
            recommended_row[max_features-1] = recommended_mse
        stepwise_df.loc['Recommended_Features'] = recommended_row

        data_save_path = self.ML_DIR / f'final_stepwise_regression_data_{self.timestamp}.csv'
        stepwise_df.to_csv(data_save_path)

        print(f"Final stepwise regression plot saved: {save_path}")
        print(f"Final stepwise regression data saved: {data_save_path}")   
        
    def _run_baseline_analysis(self, epoch, core_num, train_test_split):
        """运行基线分析（全特征）"""
        print(f"Running baseline analysis with all {len(self.feature_names)} features...")
        
        # 创建Baseline目录
        baseline_dir = self.ML_DIR / 'Baseline_Analysis'
        os.makedirs(baseline_dir, exist_ok=True)
        
        # 训练 - 使用完整的epoch数
        print(f"Training baseline with {epoch} epochs...")
        # 修复后的代码
        X_realtest_to_pass = self.X_realtest if len(self.X_realtest) > 0 else None
        perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                    self.X_train, self.y_train, self.xgb_params,
                                    X_realtest_to_pass)
        
        # 计算特征重要性
        clf_baseline = XGBRegressor(**self.xgb_params)
        clf_baseline.fit(self.X_train, self.y_train)
        feature_importance = clf_baseline.feature_importances_
        
        # 计算相关性矩阵
        feature_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        correlation_matrix = feature_df.corr()
        
        # 生成可视化
        self.generate_prediction_scatter(perf, self.feature_names, "Baseline", baseline_dir)
        
        # 生成特征重要性图
        self._plot_feature_importance(feature_importance, baseline_dir)
        
        # 生成相关性热图
        self._plot_correlation_matrix(correlation_matrix, baseline_dir)
        
        # 生成SHAP图（选择重要的特征）
        importance_indices = np.argsort(feature_importance)[-20:]  # Top 20特征
        for idx in importance_indices:
            self.generate_shap_plots(
                self.X_train[:, idx],
                perf['shap_values'][:, idx],
                perf['shap_std'][:, idx],
                self.feature_names[idx],
                baseline_dir
            )
        
        # 保存结果
        self.save_comprehensive_results(baseline_dir, "Baseline_All",
                                       self.feature_names.tolist(),
                                       list(range(len(self.feature_names))),
                                       self.X_train, perf)
        
        # 保存全特征矩阵
        self._save_full_feature_matrix(baseline_dir)
        
        results = {
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'feature_importance': feature_importance,
            'feature_names': self.feature_names
        }
        
        print(f"Baseline Results: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, R²={perf['r2']:.4f}")
        print(f"Baseline analysis used {epoch} epochs as specified")
        
        return results

    def _plot_feature_importance(self, feature_importance, save_dir):
        """绘制特征重要性图"""
        plt.figure(figsize=(15, 10))
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # 只取前15个特征
        n_features = min(15, len(importance_df))
        importance_df_top = importance_df.tail(n_features)
        
        plt.barh(range(len(importance_df_top)), importance_df_top['importance'])
        plt.yticks(range(len(importance_df_top)), importance_df_top['feature'], fontsize=20)
        plt.xlabel('Feature Importance', fontsize=24)
        plt.title('XGBoost Feature Importance (Top 15 Features)', fontsize=28)
        plt.xticks(fontsize=20)
        plt.tight_layout()
        
        # 保存图片
        importance_save_path = save_dir / f'baseline_feature_importance_{self.timestamp}.png'
        plt.savefig(importance_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始数据
        importance_data_path = save_dir / f'baseline_feature_importance_data_{self.timestamp}.csv'
        importance_df.to_csv(importance_data_path, index=False)
        
        print(f"Feature importance plot saved: {importance_save_path}")
        print(f"Feature importance data saved: {importance_data_path}")

    def _plot_correlation_matrix(self, correlation_matrix, save_dir):
        """绘制相关性热图"""
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, 
                   xticklabels=correlation_matrix.columns,
                   yticklabels=correlation_matrix.columns,
                   cmap='RdBu',
                   center=0,
                   annot=False,
                   square=True)
        plt.title('Feature Correlation Matrix', fontsize=24)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        correlation_save_path = save_dir / f'correlation_matrix_{self.timestamp}.png'
        plt.savefig(correlation_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始数据
        correlation_data_path = save_dir / f'correlation_matrix_data_{self.timestamp}.csv'
        correlation_matrix.to_csv(correlation_data_path)
        
        print(f"Correlation matrix plot saved: {correlation_save_path}")
        print(f"Correlation matrix data saved: {correlation_data_path}")

    def _save_full_feature_matrix(self, save_dir):
        """保存全特征矩阵"""
        full_feature_path = save_dir / f'full_feature_matrix_{self.timestamp}.csv'
        full_feature_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        full_feature_df['Target'] = self.y_train
        full_feature_df['Sample_Name'] = self.sample_names_train
        full_feature_df['SMILES'] = self.smiles_train
        full_feature_df.to_csv(full_feature_path, index=False)
        
        print(f"Full feature matrix saved: {full_feature_path}")
    
    def _run_stepwise_regression(self, max_features, n_runs, epoch, core_num, train_test_split):
        """运行多轮逐步回归"""
        all_results = []
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}...")
            
            # 创建Run目录
            run_dir = self.ML_DIR / f'Run_{run + 1}'
            os.makedirs(run_dir, exist_ok=True)
            
            # 单轮逐步回归
            result = self._single_stepwise_regression(max_features, epoch, core_num, 
                                                     train_test_split, run_dir, run + 1)
            result['run_id'] = run + 1
            all_results.append(result)
            
            print(f"  Run {run + 1} Final MSE: {result['final_mse']:.4f}")
        
        # 按MSE排序
        all_results.sort(key=lambda x: x['final_mse'])
        
        return all_results
    
    def _single_stepwise_regression(self, max_features, epoch, core_num, train_test_split, run_dir, run_id):
        """单次逐步回归，生成详细输出"""
        selected_indices = []
        mse_history = []
        feature_names_history = []
        
        # 找最佳单特征，使用固定种子但加入run_id以产生不同结果
        np.random.seed(42 + run_id)
        best_single_mse = np.inf
        best_single_idx = None
        
        print(f"  Finding best single feature...")
        for j in range(len(self.feature_names)):
            X_temp = self.X_train[:, [j]]
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_temp, self.y_train, self.xgb_params)
            if perf['mse'] < best_single_mse:
                best_single_mse = perf['mse']
                best_single_idx = j
        
        selected_indices.append(best_single_idx)
        mse_history.append(best_single_mse)
        feature_names_history.append([self.feature_names[best_single_idx]])
        
        print(f"    Best single feature: {self.feature_names[best_single_idx]} (MSE: {best_single_mse:.4f})")
        
        # 逐步添加特征
        for step in range(1, max_features):
            print(f"  Adding feature {step + 1}/{max_features}...")
            best_mse = np.inf
            best_feature = None
            
            remaining = [i for i in range(len(self.feature_names)) if i not in selected_indices]
            
            for feat_idx in remaining:
                candidate = selected_indices + [feat_idx]
                X_temp = self.X_train[:, candidate]
                perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                             X_temp, self.y_train, self.xgb_params)
                
                if perf['mse'] < best_mse:
                    best_mse = perf['mse']
                    best_feature = feat_idx
            
            if best_feature is not None:
                selected_indices.append(best_feature)
                mse_history.append(best_mse)
                feature_names_history.append([self.feature_names[i] for i in selected_indices])
                print(f"    Added {self.feature_names[best_feature]} (MSE: {best_mse:.4f})")
        
        # 为每个特征数量生成散点图
        for i, (mse, features) in enumerate(zip(mse_history, feature_names_history)):
            indices = selected_indices[:i+1]
            X_temp = self.X_train[:, indices]
            X_realtest_temp = self.X_realtest[:, indices] if len(self.X_realtest) > 0 else None
            
            # 完整训练以获得准确结果
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_temp, self.y_train, self.xgb_params, X_realtest_temp)
            
            # 生成散点图
            self.generate_prediction_scatter(perf, features, f"Run_{run_id}_{i+1}feats", run_dir)
        
        # 绘制MSE曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.title(f'Run {run_id}: MSE vs Number of Features', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存MSE曲线图
        mse_curve_path = run_dir / f'Run_{run_id}_MSE_curve.png'
        plt.savefig(mse_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存MSE曲线原始数据
        mse_data = pd.DataFrame({
            'Number_of_Features': range(1, len(mse_history) + 1),
            'MSE': mse_history,
            'Features': [', '.join(feature_names_history[i]) for i in range(len(mse_history))]
        })
        mse_data_path = run_dir / f'Run_{run_id}_MSE_curve_data.csv'
        mse_data.to_csv(mse_data_path, index=False)
        
        return {
            'selected_indices': selected_indices,
            'feature_names': [self.feature_names[i] for i in selected_indices],
            'mse_history': mse_history,
            'final_mse': mse_history[-1],
            'run_dir': run_dir
        }
    
    def _run_shap_analysis(self, stepwise_results, epoch, core_num, train_test_split):
        """对每个Run进行SHAP分析"""
        shap_results = []
        
        for result in stepwise_results:
            run_id = result['run_id']
            run_dir = result['run_dir']
            selected_indices = result['selected_indices']
            selected_features = result['feature_names']
            
            print(f"\nAnalyzing SHAP for Run {run_id}...")
            
            # 创建SHAP分析目录
            shap_dir = run_dir / 'SHAP_Analysis'
            os.makedirs(shap_dir, exist_ok=True)
            
            # 使用选择的特征进行训练
            X_selected = self.X_train[:, selected_indices]
            X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None
            
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_selected, self.y_train, self.xgb_params,
                                         X_realtest_selected)
            
            # 生成SHAP图
            for i, feat_name in enumerate(selected_features):
                self.generate_shap_plots(
                    X_selected[:, i],
                    perf['shap_values'][:, i],
                    perf['shap_std'][:, i],
                    feat_name,
                    shap_dir
                )
            
            # 保存详细结果
            self.save_comprehensive_results(shap_dir, f"Run_{run_id}_SHAP",
                                          selected_features, selected_indices,
                                          X_selected, perf)
            
            shap_results.append({
                'run_id': run_id,
                'features': selected_features,
                'mse': perf['mse'],
                'r2': perf['r2']
            })
        
        return shap_results
    
    def _intelligent_recommendation(self, stepwise_results, baseline_results):
        """智能推荐算法"""
        feature_scores = {}
        
        # 从逐步回归中收集分数
        for result in stepwise_results:
            for i, feat_name in enumerate(result['feature_names']):
                if feat_name not in feature_scores:
                    feature_scores[feat_name] = {'stepwise': 0, 'importance': 0, 'count': 0}
                
                feature_scores[feat_name]['stepwise'] += 1.0 / (i + 1)
                feature_scores[feat_name]['count'] += 1
        
        # 加入基线特征重要性
        if baseline_results:
            importance = baseline_results['feature_importance']
            max_importance = np.max(importance) if np.max(importance) > 0 else 1
            
            for i, feat_name in enumerate(baseline_results['feature_names']):
                if feat_name in feature_scores:
                    feature_scores[feat_name]['importance'] = importance[i] / max_importance
        
        # 计算综合分数
        recommendations = []
        for feat_name, scores in feature_scores.items():
            # 权重：逐步回归40%，特征重要性40%，出现次数20%
            stepwise_score = scores['stepwise'] / len(stepwise_results)
            importance_score = scores['importance']
            appearance_score = scores['count'] / len(stepwise_results)
            
            final_score = 0.4 * stepwise_score + 0.4 * importance_score + 0.2 * appearance_score
            
            feat_idx = np.where(self.feature_names == feat_name)[0][0]
            
            recommendations.append({
                'feature_name': feat_name,
                'feature_index': feat_idx,
                'score': final_score,
                'stepwise_score': stepwise_score,
                'importance_score': importance_score,
                'appearance_count': scores['count']
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # 保存推荐结果
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_csv(self.ML_DIR / f'feature_recommendations_{self.timestamp}.csv', index=False)
        
        print(f"\nTop recommended features:")
        for i, rec in enumerate(recommendations[:10]):
            print(f"  {i+1}. {rec['feature_name']} (score: {rec['score']:.3f})")
        
        return recommendations
    
    def _run_final_analysis(self, top_features, epoch, core_num, train_test_split):
        """使用推荐特征进行最终分析"""
        # 创建最终分析目录
        final_dir = self.ML_DIR / 'Final_Manual_Analysis'
        os.makedirs(final_dir, exist_ok=True)
        
        # 获取特征索引
        selected_indices = []
        for feat_name in top_features:
            idx = np.where(self.feature_names == feat_name)[0][0]
            selected_indices.append(idx)
        
        print(f"Running final analysis with {len(top_features)} recommended features...")
        
        # 准备数据
        X_selected = self.X_train[:, selected_indices]
        X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None
        
        # 训练
        perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                     X_selected, self.y_train, self.xgb_params,
                                     X_realtest_selected)
        
        # 生成可视化
        self.generate_prediction_scatter(perf, top_features, "Final_Manual", final_dir)
        
        # 生成SHAP分析
        for i, feat_name in enumerate(top_features):
            self.generate_shap_plots(
                X_selected[:, i],
                perf['shap_values'][:, i],
                perf['shap_std'][:, i],
                feat_name,
                final_dir
            )
        
        # 保存详细结果
        self.save_comprehensive_results(final_dir, f"Final_{len(top_features)}feats",
                                       top_features, selected_indices,
                                       X_selected, perf)
        
        # 计算测试集性能
        test_performance = None
        if len(self.y_realtest) > 0 and perf['y_realtest_pred'] is not None:
            test_mae = mean_absolute_error(self.y_realtest, perf['y_realtest_pred'])
            test_mse = mean_squared_error(self.y_realtest, perf['y_realtest_pred'])
            test_r2 = r2_score(self.y_realtest, perf['y_realtest_pred'])
            test_performance = {
                'mae': test_mae,
                'mse': test_mse,
                'r2': test_r2
            }
        
        results = {
            'selected_features': top_features,
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'test_performance': test_performance
        }
        
        print(f"Final Results: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, R²={perf['r2']:.4f}")
        
        return results
    
    def _save_workflow_summary(self, all_results, epoch, core_num, train_test_split, n_runs, max_features):
        """保存包含详细运行参数的工作流摘要"""
        summary_path = self.ML_DIR / f'workflow_summary_{self.timestamp}.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('WORKFLOW ANALYSIS SUMMARY\n')
            f.write('='*80 + '\n\n')

            # --- 新增部分：常规信息和运行参数 ---
            f.write('GENERAL & RUN PARAMETERS:\n')
            f.write('-'*50 + '\n')
            f.write(f'Analysis Timestamp : {self.timestamp}\n')
            f.write(f'Input Data File    : {self.data_file}\n')
            f.write(f'Output Directory   : {self.ML_DIR}\n')
            f.write(f'Epochs per Training: {epoch}\n')
            f.write(f'CPU Cores Used     : {core_num}\n\n')

            # --- 新增部分：数据集和特征选择参数 ---
            f.write('DATASET & FEATURE SELECTION PARAMETERS:\n')
            f.write('-'*50 + '\n')
            f.write(f'Train/Test Split Ratio : {train_test_split}\n')
            f.write(f'Training Samples       : {self.dataset_info["train_samples"]}\n')
            f.write(f'Test Samples           : {self.dataset_info["test_samples"]}\n')
            if self.test_sample_names:
                # 只显示找到的样本，更简洁
                f.write(f'Specified Test Samples : {self.dataset_info["test_sample_names_found"]}\n')
            else:
                f.write('Specified Test Samples : None (Random Split)\n')
            f.write(f'Stepwise Reg. Runs (n_runs)      : {n_runs}\n')
            f.write(f'Max Features per Run (max_features): {max_features}\n\n')

            # --- 原有部分：分析结果 ---
            f.write('BASELINE ANALYSIS (ALL FEATURES):\n')
            f.write('-'*50 + '\n')
            f.write(f"MSE: {all_results['baseline']['mse']:.4f}\n")
            f.write(f"R²: {all_results['baseline']['r2']:.4f}\n\n")

            f.write('STEPWISE REGRESSION RESULTS:\n')
            f.write('-'*50 + '\n')
            for result in all_results['stepwise']:
                f.write(f"Run {result['run_id']}: MSE = {result['final_mse']:.4f}\n")
                f.write(f"  Features: {', '.join(result['feature_names'])}\n")
            f.write('\n')

            f.write(f'TOP {max_features} RECOMMENDED FEATURES:\n')
            f.write('-'*50 + '\n')
            # 使用 max_features 来决定显示多少个推荐特征
            for i, rec in enumerate(all_results['recommendations'][:max_features]):
                f.write(f"{i+1}. {rec['feature_name']} (score: {rec['score']:.3f})\n")
            f.write('\n')

            f.write('FINAL ANALYSIS RESULTS (WITH RECOMMENDED FEATURES):\n')
            f.write('-'*50 + '\n')
            f.write(f"Selected features: {', '.join(all_results['final']['selected_features'])}\n")
            f.write(f"Cross-Validation MSE: {all_results['final']['mse']:.4f}\n")
            f.write(f"Cross-Validation R²: {all_results['final']['r2']:.4f}\n")

            if all_results['final'].get('test_performance'):
                f.write(f"Test Set MSE: {all_results['final']['test_performance']['mse']:.4f}\n")
                f.write(f"Test Set R²: {all_results['final']['test_performance']['r2']:.4f}\n")

            f.write('\n' + '='*80 + '\n')
            f.write('WORKFLOW ANALYSIS COMPLETED SUCCESSFULLY\n')
            f.write('='*80 + '\n')

# ================== 主工作流控制器 ==================
class ChemMLWorkflow:
    """
    化学机器学习工作流控制器
    """
    
    @staticmethod
    def run_analysis(mode, data_file, test_sample_names=None, nan_handling='drop_columns',
                     output_dir=None, features=None, **kwargs):
        """
        运行分析的统一入口
        
        Args:
            mode: 'manual' 或 'workflow'
            data_file: 数据文件路径
            test_sample_names: 测试集样本名称
            nan_handling: NaN处理方式
            output_dir: 输出目录
            features: 手动模式下的特征列表，或 'Full' 表示所有特征
            **kwargs: 其他参数（epoch, core_num等）
        
        Returns:
            dict: 分析结果
        """
        
        # 提取通用参数
        epoch = kwargs.get('epoch', 64)
        core_num = kwargs.get('core_num', 32)
        train_test_split = kwargs.get('train_test_split', 0.85)
        
        if mode == 'manual':
            # 手动特征分析模式
            print("="*60)
            print("RUNNING MANUAL FEATURE ANALYSIS")
            print("="*60)
            
            if not features:
                raise ValueError("Manual mode requires 'features' parameter")
            
            analyzer = ManualFeatureAnalyzer(data_file, test_sample_names, nan_handling, 
                                            output_dir, features)
            generate_fitting = kwargs.get('generate_fitting', True)
            results = analyzer.run(epoch, core_num, train_test_split, generate_fitting)
            
            return results
            
        elif mode == 'workflow':
            # 完整工作流模式
            print("="*60)
            print("RUNNING COMPLETE WORKFLOW")
            print("="*60)
            
            max_features = kwargs.get('max_features', 5)
            n_runs = kwargs.get('n_runs', 3)
            
            analyzer = WorkflowAnalyzer(data_file, test_sample_names, nan_handling, output_dir)
            results = analyzer.run(max_features, n_runs, epoch, core_num, train_test_split)
            
            return results
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'manual' or 'workflow'")


# ================== 使用示例 ==================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Example: Complete Workflow Analysis")
    print("="*80)
    
    results_workflow = ChemMLWorkflow.run_analysis(
        mode='workflow',
        data_file="/home/yumingsu/Python/Project_wangqi/250818/final/Surfacia_3.0_20250820_102612/FinalFull_Mode3_97_86.csv",
        test_sample_names=[2, 31, 35, 39] + list(range(41, 97)),
        nan_handling='drop_columns',
        max_features=6,
        n_runs=3,
        epoch=128,
        core_num=32
    )
    
    print(f"\nWorkflow completed.")
    print(f"Baseline MSE: {results_workflow['baseline']['mse']:.4f}")
    print(f"Final MSE: {results_workflow['final']['mse']:.4f}")
    print(f"Selected features: {results_workflow['final']['selected_features']}")