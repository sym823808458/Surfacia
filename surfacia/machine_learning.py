# surfacia/machine_learning.py

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import shap
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from multiprocessing import Pool
import joblib
import pandas as pd

def generate_feature_matrix(merged_output_filename, output_dir, nan_handling='drop_rows'):
    """
    Generate feature matrix and related files from the merged CSV file.
    
    Args:
        merged_output_filename (str): Path to the merged CSV file.
        output_dir (str): Path to the output directory.
        nan_handling (str): How to handle NaN values. Options are 'drop_rows' or 'drop_columns'.
    
    Returns:
        dict: A dictionary containing the paths of the generated files.
    """
    merged_df = pd.read_csv(merged_output_filename)
    
    # Handle NaN values
    if nan_handling == 'drop_rows':
        merged_df = merged_df.dropna()
    elif nan_handling == 'drop_columns':
        merged_df = merged_df.dropna(axis=1)
    else:
        raise ValueError("nan_handling must be either 'drop_rows' or 'drop_columns'")
    
    S_N = merged_df.shape[0]
    F_N = merged_df.shape[1] - 3  # Assuming 'Sample Name', 'smiles', 'target' are three columns

    # Create a MachineLearning folder inside the specified output directory
    ML_DIR = Path(output_dir, "MachineLearning")
    ML_DIR.mkdir(parents=True, exist_ok=True)

    # Save SMILES
    INPUT_SMILES = Path(ML_DIR, f'Smiles_{S_N}.csv')
    merged_df[['smiles']].to_csv(INPUT_SMILES, index=False, header=False)

    # Save labels (values)
    INPUT_Y = Path(ML_DIR, f'Values_True_{S_N}.csv')
    merged_df[['target']].to_csv(INPUT_Y, index=False, header=False)

    # Save feature matrix
    INPUT_X = Path(ML_DIR, f'Features_{S_N}_{F_N}.csv')
    merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1).to_csv(INPUT_X, index=False, float_format='%.6f', header=False)

    # Save feature names with spaces replaced by underscores
    INPUT_TITLE = Path(ML_DIR, f'Title_{F_N}.csv')
    with open(INPUT_TITLE, 'w') as f:
        for col in merged_df.columns[1:-2]:
            # Replace spaces and slashes with underscores
            formatted_col = col.replace(' ', '_').replace('/', '_')
            f.write(formatted_col + '\n')

    print(f'Machine Learning data split and saved in {ML_DIR}:')
    print(f'  - SMILES: {INPUT_SMILES.name}')
    print(f'  - Values: {INPUT_Y.name}')
    print(f'  - Features: {INPUT_X.name}')
    print(f'  - Feature Titles: {INPUT_TITLE.name}')
    print("All processing completed.")

    return {
        'smiles': str(INPUT_SMILES),
        'values': str(INPUT_Y),
        'features': str(INPUT_X),
        'titles': str(INPUT_TITLE),
        'ml_dir': str(ML_DIR)
    }
    
    
def XGB_Fit(X, y, X_train, y_train, X_test, y_test, paras):
    clf_new = XGBRegressor()
    for k, v in paras.items():
        clf_new.set_params(**{k: v})
    # Fit model
    clf_new.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    # Compute Loss
    y_pred = clf_new.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    shap_values = shap.TreeExplainer(clf_new).shap_values(X)
    s = np.mean(clf_new.predict(X)) - np.mean(y_train)
    s2 = np.mean(clf_new.predict(X)) - np.mean(y)
    #print(np.sum(shap_values), s, s2)
    #print(' MSE: %.5f' % mse, ' MAE: %.5f' % mae, ' R^2: %.5f' % r2)
    return [mse, mae, r2, shap_values, clf_new]
    
def xgb_stepwise_regression(
    input_x,
    input_y,
    input_title,
    
    epoch=32,
    core_num=32,
    train_test_split_ratio=0.85,
    step_feat_num=2,
    know_ini_feat=False,
    ini_feat=[],
    test_indices=[],
    output_dir=None 
):
    # Get current time
    c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    c_time_m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    version='V2.2'
    # Parameters
    EPOCH = epoch
    CORE_NUM = core_num
    TRAIN_TEST_SPLIT = train_test_split_ratio

    # Load data
    X = np.loadtxt(input_x, delimiter=',')
    y = np.loadtxt(input_y)
    title = np.loadtxt(input_title, dtype=str, delimiter=',', comments='!')

    S_N = X.shape[0]  # Number of samples
    F_N = X.shape[1]  # Number of features

    # Initialize first feature
    KNOW_inifeat = know_ini_feat
    inifeat = ini_feat

    # Step feature number
    Stepfeatnum = step_feat_num

    # Adjust test_indices
    if not test_indices:
        test_indices = []
    else:
        test_indices = [i - 1 for i in test_indices]  # Adjust index if necessary

    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_indices] = True
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_realtest = X[test_mask]
    y_realtest = y[test_mask]
    # Update X and y to be training data
    X = X_train
    y = y_train

    # Create directory and record file
    if output_dir is None:
        output_dir = '.'
    
    DIR_name = 'Stepreg-XGB_' + version + '_' + c_time
    DIR = Path(output_dir, DIR_name)
    os.makedirs(DIR, exist_ok=True)
    RECORD_NAME = 'Record_Stepreg_' + version + '_' + c_time + '.txt'
    RECORD_NAME = Path('.', DIR, RECORD_NAME)
    f1 = open(RECORD_NAME, 'w')
    f1.write('Record of XGB-Stepregression ' + version + '\n\n')
    f1.write('Generation time: ' + c_time_m + '\n\n\n')
    f1.write('Input files are: ' + input_x + ', ' + input_y + ', ' + input_title + '\n\n\n')
    f1.write('EPOCH= ' + str(EPOCH) + ' CORE_NUM= ' + str(CORE_NUM) + ' split_ratio= ' + str(round(TRAIN_TEST_SPLIT, 3)) + '\n\n\n')
    f1.write('Test indices in feature matrix: ' + str(test_indices) + '\n\n\n')

    # Save training and test data
    save_nameX = 'Feature_X_train_' + str(len(y)) + '_' + str(F_N) + '_' + c_time + '.csv'
    save_nameX = Path('.', DIR, save_nameX)
    np.savetxt(save_nameX, X_train, fmt='%s', delimiter=',')

    save_nameX = 'Feature_X_realtest_' + str(len(y_realtest)) + '_' + str(F_N) + '_' + c_time + '.csv'
    save_nameX = Path('.', DIR, save_nameX)
    np.savetxt(save_nameX, X_realtest, fmt='%s', delimiter=',')

    save_nameY = 'Value_y_train_' + str(len(y)) + '_' + c_time + '.csv'
    save_nameY = Path('.', DIR, save_nameY)
    np.savetxt(save_nameY, y_train, fmt='%s', delimiter=',')

    save_nameY = 'Value_y_realtest_' + str(len(y_realtest)) + '_' + c_time + '.csv'
    save_nameY = Path('.', DIR, save_nameY)
    np.savetxt(save_nameY, y_realtest, fmt='%s', delimiter=',')

    # Initialize model parameters
    clf = XGBRegressor(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=8,
        verbosity=0,
        booster='gbtree',
        reg_alpha=np.exp(-3),
        reg_lambda=np.exp(-3),
        gamma=np.exp(-5),
        subsample=0.5,
        objective='reg:squarederror',
        n_jobs=1
    )
    paras = clf.get_params()


    def poolfit(TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, X, y, paras, X_realtest=None):
        r_l = []
        split_l = []
        test_idx_m = []
        point = round(X.shape[0] * TRAIN_TEST_SPLIT)
        for _ in range(int(EPOCH / CORE_NUM)):
            print('Round', int(CORE_NUM * _) + 1, 'Begin:')
            pool = Pool(CORE_NUM)
            for __ in range(CORE_NUM):
                permutation = np.random.permutation(y.shape[0])
                train_idx = permutation[:point]
                test_idx = permutation[point:]
                X_train = X[train_idx, :]
                y_train = y[train_idx]
                X_test = X[test_idx, :]
                y_test = y[test_idx]
                split_l.append(train_idx)
                test_idx_m.append(test_idx)
                r = pool.apply_async(XGB_Fit, args=(X, y, X_train, y_train, X_test, y_test, paras,))
                r_l.append(r)
            pool.close()
            pool.join()

        mse_list = []
        mae_list = []
        r2_list = []
        shap_m = np.zeros((X.shape[0], X.shape[1]))
        full_m = np.zeros((len(r_l), X.shape[0]))
        y_realtest_pred_list = []
        for i in range(len(r_l)):
            r = r_l[i]
            results = r.get()
            temp = results
            mse = temp[0]
            mae = temp[1]
            r2 = temp[2]
            shap_values = temp[3]
            clf_new = temp[4]
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
            shap_m += shap_values
            train_idx = split_l[i]
            test_idx = []
            for j in range(X.shape[0]):
                if j not in train_idx:
                    test_idx.append(j)
            y_full_pred = clf_new.predict(X)
            full_m[i] = y_full_pred
            if X_realtest is not None:
                y_realtest_pred = clf_new.predict(X_realtest)
                y_realtest_pred_list.append(y_realtest_pred)
        mse1 = np.mean(mse_list)
        mae1 = np.mean(mae_list)
        r21 = np.mean(r2_list)
        mse2 = np.std(mse_list)
        shap_m2 = shap_m / len(r2_list)
        y_realtest_pred_std = np.std(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
        y_realtest_pred_mean = np.mean(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
        return [mse1, mae1, r21, full_m, test_idx_m, mse2, shap_m2, y_realtest_pred_mean, y_realtest_pred_std]

    # Start stepwise feature selection
    if not KNOW_inifeat:
        perflist1 = []
        for j in range(F_N):
            print('Round', j)
            f1.write(f'Round {j} - Initial Feature Selection\n')
            inifeat = title[j]
            inifeatindex = np.where(title == inifeat)[0][0]
            featlist = [inifeatindex]
            Xtemp = X[:, featlist]
            perf = poolfit(TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, Xtemp, y, paras)
            perflist1.append(perf[0])
        inifeat = np.argmin(perflist1)
        print(f'Best initial feature index: {inifeat}')
        f1.write(f'Best initial feature index: {inifeat}\n')
        print(f'Minimum MSE: {np.min(perflist1)}')
        f1.write(f'Minimum MSE: {np.min(perflist1)}\n')
        perflistt = np.argsort(perflist1)
        print('Top 10 features:')
        f1.write('Top 10 initial features:\n')
        for _ in range(10):
            print(title[perflistt[_]])
            f1.write(f'{title[perflistt[_]]}\n')
        inifeatindex = inifeat
    else:
        print(f'Initial feature provided: {title[inifeat]}')
        f1.write(f'Initial feature provided: {title[inifeat]}\n')
        inifeatindex = ini_feat[0] if ini_feat else 0

    # Stepwise feature addition
    print('Starting stepwise feature selection...')
    f1.write('Starting stepwise feature selection...\n')
    featlist = []
    bestfeatlist = []
    mselist = []
    for i in range(Stepfeatnum):
        perflist = np.full(len(title), np.inf)
        print(f'Now selecting {i+1} features. Beginning regression!')
        f1.write(f'Now selecting {i+1} features. Beginning regression!\n')
        if i == 0:
            featlist.append(inifeatindex)
            Xtemp = X[:, featlist]
            perf = poolfit(TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, Xtemp, y, paras)
            perflist[inifeatindex] = perf[0]
            bestfeatlist = featlist.copy()
            print(f'Round 1 - Features: {title[bestfeatlist]} MSE: {perflist[inifeatindex]}\n')
            f1.write(f'Round 1 - Features: {title[bestfeatlist]} MSE: {perflist[inifeatindex]}\n')
            mselist.append(perflist[inifeatindex])
        else:
            for j in range(F_N):
                if j in bestfeatlist:
                    continue  # Skip already selected features
                featlist = bestfeatlist.copy()
                featlist.append(j)
                Xtemp = X[:, featlist]
                perf = poolfit(TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, Xtemp, y, paras)
                perflist[j] = perf[0]
            min_mse = np.min(perflist)
            mseind = np.argmin(perflist)
            print(f'Selected feature index: {mseind}')
            f1.write(f'Selected feature index: {mseind}\n')
            bestfeatlist.append(mseind)
            round_info = (f'Round {len(bestfeatlist)} - Features: {title[bestfeatlist]} MSE: {round(min_mse, 4)}\n')
            print(round_info)
            f1.write(round_info)
            mselist.append(min_mse)

    # Close the log file
    f1.close()

    # Save best features
    save_name1 = 'XGB-Stepregression_best_titles_' + str(len(bestfeatlist)) + '_' + c_time + '.txt'
    save_name2 = Path('.', DIR, save_name1)
    np.savetxt(save_name2, title[bestfeatlist], fmt='%s', delimiter=',', comments='!')

    # Plot MSE vs Feature Number
    x = np.arange(1, len(mselist) + 1)
    y = mselist
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs Number of Features')
    plt.savefig(Path(DIR, 'MSE_vs_Features_' + c_time + '.png'))
    plt.close()

    # Final model evaluation with selected features
    X = np.loadtxt(input_x, delimiter=',')
    y = np.loadtxt(input_y)
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_realtest = X[test_mask]
    y_realtest = y[test_mask]
    X = X_train
    y = y_train
    print('X:', X.shape, ' y:', y.shape)

    besttitle = title[bestfeatlist]
    titlelist = []
    mselist = []
    for jj in range(len(besttitle)):
        thistitleindex = np.where(title == besttitle[jj])[0][0]
        titlelist.append(thistitleindex)
        Xtemp = X[:, titlelist]
        X_realtest_temp = X_realtest[:, titlelist]
        perf = poolfit(TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, Xtemp, y, paras, X_realtest_temp)
        y_realtest_pred = perf[7]
        y_realtest_pred_std = perf[8]

        # Save real test predictions
        append_filename = Path('.', DIR, 'All_Realtestpred_' + c_time + '.txt')
        with open(append_filename, 'a+') as f:
            np.savetxt(f, y_realtest_pred.reshape(1, -1), fmt='%s', delimiter=',')

        # Save real test prediction std
        append_filename = Path('.', DIR, 'All_Realtestpredstd_' + c_time + '.txt')
        with open(append_filename, 'a+') as f:
            np.savetxt(f, y_realtest_pred_std.reshape(1, -1), fmt='%s', delimiter=',')

        full_m = perf[3]
        full_m = np.array(full_m)
        test_idx_m = perf[4]
        test_idx_m = np.array(test_idx_m)
        save_name = 'XGBoost_Test_Index_' + str(len(Xtemp[0])) + '_' + c_time + '.csv'
        save_name = Path('.', DIR, save_name)
        np.savetxt(save_name, test_idx_m, fmt='%d', delimiter=',')
        shap_m = perf[6]
        save_name = 'SHAP_Matrix_' + str(len(titlelist)) + '_' + c_time + '.csv'
        save_name = Path('.', DIR, save_name)
        np.savetxt(save_name, shap_m, fmt='%s', delimiter=',')
        save_nameX = 'Feature_Matrix_' + str(len(titlelist)) + '_' + c_time + '.csv'
        save_nameX = Path('.', DIR, save_nameX)
        np.savetxt(save_nameX, Xtemp, fmt='%s', delimiter=',')

        # Compute test predictions
        test_data_m = [[] for _ in range(X.shape[0])]
        for i in range(test_idx_m.shape[0]):
            for j in range(test_idx_m.shape[1]):
                test_data_m[test_idx_m[i, j]].append(full_m[i, test_idx_m[i, j]])
        test_mean_l = [np.mean(test_data_m[i]) for i in range(X.shape[0])]
        test_std_l = [np.std(test_data_m[i]) for i in range(X.shape[0])]

        # Plot scatter of mean predictions
        true_y = y.flatten().tolist()
        plt.figure(figsize=(10, 8), dpi=300)
        sc = plt.scatter(true_y, test_mean_l, alpha=0.55, c=test_std_l, cmap='viridis', marker='o')
        left_limit = min(min(true_y) - 1, min(test_mean_l) - 1)
        right_limit = max(max(true_y) + 1, max(test_mean_l) + 1)
        plt.plot([left_limit, right_limit], [left_limit, right_limit], color='#B22222', linestyle=':', linewidth=2)
        plt.plot([left_limit, right_limit], [left_limit + 1, right_limit + 1], color='#FFA500', linestyle=':', linewidth=2)
        plt.plot([left_limit, right_limit], [left_limit - 1, right_limit - 1], color='#FFA500', linestyle=':', linewidth=2)
        plt.legend(['Perfect Fit', '+1', '-1', 'Mean of Test Predictions'], loc='upper left', fontsize=17, shadow=True)
        plt.xlabel('True Values', fontsize=17)
        plt.ylabel('Mean Predicted Values', fontsize=17)
        plt.title('Scatter of Mean Test Prediction vs True Values with ' + str(len(Xtemp[0])) + ' Top Features\n' +
                  'Mean Test:  MSE: ' + str(round(perf[0], 4)) +
                  '  MAE: ' + str(round(perf[1], 4)) +
                  '  R^2: ' + str(round(perf[2], 4)), fontsize=21)
        cb = plt.colorbar(sc)
        cb.set_label('Std of Test Predictions', fontsize=17)
        plt.grid(which='major', color='#D5D5D5', alpha=0.5)
        save_name = 'XGBoost_Mean_Test_Prediction_Scatter_' + str(len(Xtemp[0])) + '_' + c_time + '.png'
        save_name = Path('.', DIR, save_name)
        plt.savefig(save_name)
        plt.close()

        # Save scatter plot data
        xgbscatterdata = np.column_stack((true_y, test_mean_l, test_std_l))
        save_name1 = ('XGBoost_Mean_Test_Prediction_Data_' + str(len(Xtemp[0])) + '_TopFeatures' +
                      '_MSE' + str(round(perf[0], 3)) +
                      '_MSEstd' + str(round(perf[5], 3)) +
                      '_MAE' + str(round(perf[1], 3)) +
                      '_R2' + str(round(perf[2], 3)) + '_' + besttitle[jj] + '.txt')
        save_name2 = Path('.', DIR, save_name1)
        np.savetxt(save_name2, xgbscatterdata, fmt='%s', delimiter=',', comments='!')

        mselist.append(round(perf[0], 3))

    # Find the best feature set
    best_mse_index = mselist.index(min(mselist))
    print('Best MSE List:', mselist)
    print('Best MSE Index:', best_mse_index)
    print('Best Feature Title List:', title[bestfeatlist])

    # Generate SHAP scatter plots
    for jj in range(len(bestfeatlist)):
        plt.figure(figsize=(10, 8), dpi=300)
        plt.scatter(Xtemp[:, jj], shap_m[:, jj])
        plt.xlabel('Feature Values')
        plt.ylabel('SHAP Values')
        plt.title('SHAP Scatter Plot of ' + title[bestfeatlist[jj]])
        save_name = 'SHAP_Scatter_' + str(title[bestfeatlist[jj]]) + '_' + c_time + '.png'
        save_name = Path('.', DIR, save_name)
        plt.savefig(save_name)
        plt.close()

    print('Machine learning process completed successfully.')
