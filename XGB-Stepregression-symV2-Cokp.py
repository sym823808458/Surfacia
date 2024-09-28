#!/usr/bin/env python
# coding: utf-8

# ##### XGB-Stepregression   
# ##20220409sym编写 
# ### 简介：  
# 使用逐步回归的方法提取最重要的特征相互作用
# ### 更新记录：  
# V2，20221221sym更新，  
# V2.1,20240117sym更新，加入yrealtestpred，假如要预测完全数据外的情况，用这个程序
# V2.2，20240408sym更改，加入pred的std，并可以任意指定全集中的测试集，根据化学假的需求

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import os
from pathlib import Path
c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
c_time_m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# In[2]:


# 参数
# ======== System Setup ========
Version = 'V2.2sym'
EPOCH = 32
CORE_NUM = 32
# 确保EPOCH*REPEAT_ROUND是CORE_NUM的整数倍
TRAIN_TEST_SPLIT = 0.85
# ======== Fit Data Input ========
S_N = 43
F_N = 63
INPUT_X = 'Features_'+str(S_N)+'_'+str(F_N)+'.csv'
INPUT_Y = 'Values_True_'+str(S_N)+'.csv'
INPUT_TITLE = 'Title_'+str(F_N)+'.csv'
INPUT_SMILES = 'Smiles_'+str(S_N)+'.csv'
RECORD_NAME = 'Record_Stepreg_'+Version+'_'+c_time+'.txt'

X = np.loadtxt(INPUT_X, delimiter=',')
y = np.loadtxt(INPUT_Y)
title = np.loadtxt(INPUT_TITLE, dtype=str, delimiter=',', comments='!')
#是否设定第一特征
KNOW_inifeat=False
inifeat=[37]
# 记得减一
#F_N-1为全部特征都筛选
Stepfeatnum=2

# List of test set indices provided by the user
test_indices = [5]  # 对应是Full的matrix是刚刚好（有title的特征和标签值的Full）
#real_testset_len=4
#test_indices= np.random.choice(S_N, size=real_testset_len, replace=False)
test_indices = [i - 2 for i in test_indices]
test_mask = np.zeros(len(y), dtype=bool)
test_mask[test_indices] = True
X_train = X[~test_mask]
y_train = y[~test_mask]
Xrealtest = X[test_mask]
yrealtest = y[test_mask]
X=X_train
y=y_train


# In[3]:


len(y)


# In[4]:


yrealtest


# In[5]:


import shap
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
import joblib
from multiprocessing import Pool


# In[6]:


DIR = 'Stepreg-XGB_'+Version+'_'+c_time
os.mkdir(DIR)
RECORD_NAME = Path('.', DIR, RECORD_NAME)
f1 = open(RECORD_NAME, 'w')
f1.write('Record of XGB-Stepregression '+Version+'\n\n')
f1.write('Generation time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n\n\n')
f1.write('Inputfiles are: '+INPUT_X+'+'+INPUT_Y+'+'+INPUT_TITLE+'+'+'\n\n\n')
f1.write('EPOCH= '+str(EPOCH)+' CORENUM= '+str(CORE_NUM)+INPUT_X+' splitratio= '+str(round(TRAIN_TEST_SPLIT,3))+'\n\n\n')
f1.write('test list see Feature matrix '+str(test_indices)+'\n\n\n')

save_nameX = 'Feature_X_train_'+str(len(y))+'_'+str(F_N)+'_'+c_time+'.csv'
save_nameX = Path('.', DIR, save_nameX)
np.savetxt(save_nameX, X_train, fmt='%s', delimiter=',')
save_nameX = 'Feature_X_realtest_'+str(len(yrealtest))+'_'+str(F_N)+'_'+c_time+'.csv'
save_nameX = Path('.', DIR, save_nameX)
np.savetxt(save_nameX, Xrealtest, fmt='%s', delimiter=',')
save_nameX = 'Value_y_train_'+str(len(y))+'_'+c_time+'.csv'
save_nameX = Path('.', DIR, save_nameX)
np.savetxt(save_nameX, y_train, fmt='%s', delimiter=',')
save_nameX = 'Value_y_realtest_'+str(len(yrealtest))+'_'+c_time+'.csv'
save_nameX = Path('.', DIR, save_nameX)
np.savetxt(save_nameX, yrealtest, fmt='%s', delimiter=',')


# In[7]:


clf = XGBRegressor(n_estimators=350, learning_rate=0.03, max_depth=8, verbosity=0, booster='gbtree', 
                   reg_alpha=np.exp(-3), reg_lambda=np.exp(-3), gamma=np.exp(-5), 
                   subsample=0.5, objective= 'reg:squarederror', n_jobs=1)
paras = clf.get_params()
mse_list = []
mae_list = []
r2_list = []


# In[8]:


def XGB_Fit(X, y, X_train, y_train, X_test, y_test, paras):
    clf_new = XGBRegressor()
    for k, v in paras.items():
        clf_new.set_params(**{k: v})
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # 拟合模型
    clf_new.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    # 计算损失
    y_pred = clf_new.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     shap_values = shap.TreeExplainer(clf_new).shap_values(X)
#     s = np.mean(clf_new.predict(X))-np.mean(y_train)
#     s2 = np.mean(clf_new.predict(X))-np.mean(y)
#     print(np.sum(shap_values), s, s2)
#     # f_i = clf_new.feature_importances_
#      temp = [mse, mae, r2, shap_values, s, s2]
    temp = [mse]
#     print('   MSE: %.5f' % mse, '  MAE: %.5f' % mae, '  R^2: %.5f' % r2)
    del y_pred
    return (temp, 'None')
def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


# In[9]:


def poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,X, y, paras):
    r_l = []
    point = round(X.shape[0]*TRAIN_TEST_SPLIT)
    for _ in range(int(EPOCH/CORE_NUM)):
        print('Round', CORE_NUM*(_)+1, 'Begin:')
        pool = Pool(CORE_NUM)
        for __ in range(CORE_NUM):
            permutation = np.random.permutation(y.shape[0])
            train_idx = permutation[:point]
            test_idx = permutation[point:]
            X_train = X[train_idx, :]
            y_train = y[train_idx]
            X_test = X[test_idx, :]
            y_test = y[test_idx]
            r = pool.apply_async(XGB_Fit, args=(X, y, X_train, y_train, X_test, y_test, paras,))
            r_l.append(r)
        pool.close()
        pool.join()
    mse_list=[]
#     mae_list=[]
#     r2_list=[]
#     shap_m = np.zeros((S_N, F_Ntemp))
    for i in range(len(r_l)):
        r = r_l[i]
        results = r.get()
        temp = results[0]
        mse = temp[0]
#         mae = temp[1]
#         r2 = temp[2]
        mse_list.append(mse)
#         mae_list.append(mae)
#         r2_list.append(r2)

    mse1=np.mean(mse_list)
#     mae1=np.mean(mae_list)
#     r21=np.mean(r2_list)
#     temp = [mse1, mae1, r21]
    temp = [mse1]
    return temp


# In[10]:


#在这里mse越小越好
if KNOW_inifeat == False:
    perflist1=[]
    for j in range (F_N):  
        print('Round',j)
        inifeat=title[j]
        inifeatindex=np.where(title==inifeat)[0][0]
        featlist=[]
        featlist.append(inifeatindex)
        Xtemp=X[:,featlist]
        perf=poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,Xtemp, y, paras)
        perflist1.append(perf[0])    
    inifeat=np.where(perflist1==np.min(perflist1))
    print(inifeat)
    print(np.min(perflist1))
    print(perflist1)
    print(np.argsort(perflist1))
    perflistt=np.argsort(perflist1)
    for _ in range(10):
        print(title[perflistt[_]])
#画分布图

else:
    print('Already given first feature is ',title[inifeat],inifeat)


# In[11]:


title[inifeat]


# In[12]:


inifeatindex=np.where(title==title[inifeat])[0][0]
print('first feature is ',inifeatindex)
featlist=[]
bestfeatlist=[]
mseind=[]
mselist=[]
for i in range(Stepfeatnum):#for i in range(F_N-1):
    perflist=np.linspace(0,0,len(title))
    print('Now we have ', i+1, 'Features.Begin regression!')
    if i ==0:       
        featlist.append(inifeatindex)
        print(featlist)
        Xtemp=X[:,featlist]
        print(Xtemp[0])
        perf=poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,Xtemp, y, paras)
        perflist=perf[0]
        bestfeatlist=featlist
        print('Round 1_'+str(title[bestfeatlist])+'_'+str(perflist)+'\n')
        f1.write('Round 1_'+str(title[bestfeatlist])+'_'+str(perflist)+'\n')
    else:
        for j in range (F_N):     
            featlist=bestfeatlist.copy()
            print('j=',j)
            if j in bestfeatlist:
                print('Already selected feature!')
            else:                   
                featlist.append(j)
                print('featlist=',featlist)
                Xtemp=X[:,featlist]
                print(Xtemp[0])
                perf=poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,Xtemp, y, paras)
                perflist[j]=perf[0]
                print('perflist=',perflist)  
        max2 = np.sort(perflist)[i]
        mseind = np.argsort(perflist)[i]
        print('This feature is',mseind)
        print('best performance is',max2)
        bestfeatlist.append(mseind)
        print('bestfeatlist=',bestfeatlist)
        print(('Round '+str(len(title[bestfeatlist]))+'_'+str(title[bestfeatlist])+'_'+str(round(max2,4))+'\n'))
        f1.write('Round '+str(len(title[bestfeatlist]))+'_'+str(title[bestfeatlist])+'_'+str(round(max2,4))+'\n')
        mselist.append(max2)
        print(mselist)
f1.close()


# In[13]:


save_name1 = 'XGB-Stepregression_besttitles'+str(len(title[bestfeatlist]))+c_time+'.txt'
save_name2 = Path('.', DIR, save_name1)
with open(save_name2,"w") as f:
    np.savetxt(save_name2,title[bestfeatlist], fmt='%s', delimiter=',', comments='!')
f.close()
import matplotlib.pyplot as plt
import numpy as np
# mselist.insert( 0, perf0)
# make data
x = np.linspace(1, Stepfeatnum-1,Stepfeatnum-1)
print(len(mselist))
y = mselist


# In[14]:


from sklearn import model_selection
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
def poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,X, y, paras,Xrealtest):
    r_l = []
    split_l=[]
    test_idx_m=[]
    point = round(X.shape[0]*TRAIN_TEST_SPLIT)
    for _ in range(int(EPOCH/CORE_NUM)):
        print('Round', CORE_NUM*(_)+1, 'Begin:')
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

    mse_list=[]
    mae_list=[]
    r2_list=[]
    results = r.get()
    shap_m = np.zeros((S_N-len(Xrealtest), len(X[0])))
    full_m=np.zeros((len(r_l),X.shape[0]))
    y_realtest_pred_list = []
    for i in range(len(r_l)):
        r = r_l[i]
        results = r.get()
        temp = results[0]
        mse = temp[0]
        mae = temp[1]
        r2 = temp[2]
        shap_m += temp[3] 
        clf_new=results[1]
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        train_idx = split_l[i]
        test_idx = []
        for j in range(X.shape[0]):
            if j not in train_idx:
                test_idx.append(j)
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]
        y_full_pred = clf_new.predict(X)
        y_realtest_pred=clf_new.predict(Xrealtest)
#         print(y_realtest_pred)
        y_realtest_pred_list.append(y_realtest_pred)
        
#         print(y_full_pred[0])
        full_m[i]=np.array(y_full_pred)
    mse1=np.mean(mse_list)
    mae1=np.mean(mae_list)
    r21=np.mean(r2_list)
    mse2=np.std(mse_list)
    shap_m2 = shap_m/len(r2_list)
    y_realtest_pred_std = np.std(y_realtest_pred_list, axis=0)
    temp = [mse1, mae1, r21,full_m,test_idx_m,mse2,shap_m2,y_realtest_pred,y_realtest_pred_std]
    return temp
def XGB_Fit(X, y, X_train, y_train, X_test, y_test, paras):
    clf_new = XGBRegressor()
    for k, v in paras.items():
        clf_new.set_params(**{k: v})
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # 拟合模型
    clf_new.fit(X_train, y_train, eval_set=[(X_test, y_test)],  verbose=False)
    # 计算损失
    y_pred = clf_new.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    shap_values = shap.TreeExplainer(clf_new).shap_values(X)
    temp0 = [mse, mae, r2, shap_values]
    print('   MSE: %.5f' % mse, '  MAE: %.5f' % mae, '  R^2: %.5f' % r2)
    return (temp0, clf_new)


# Create clf (sklean的API)
shap_m = np.zeros((S_N, len(X[0])))
paras = clf.get_params()
mse_list = []
mae_list = []
r2_list = []
f_i = np.zeros((title.shape[0], 1))
max_r2 = -999.9


# In[15]:


title[bestfeatlist]


# In[16]:


# bestfeatlist=[8, 84, 89, 68, 109, 2]
# title[bestfeatlist]


# In[17]:


X = np.loadtxt(INPUT_X, delimiter=',')
y = np.loadtxt(INPUT_Y)
X_train = X[~test_mask]
y_train = y[~test_mask]
Xrealtest = X[test_mask]
yrealtest = y[test_mask]
X=X_train
y=y_train
print('X:', X.shape, '   y:', y.shape)
besttitle =title[bestfeatlist]
thistitleindex=[]
titlelist=[]
mselist=[]
Xtemp=[]
y_realtest_pred_list=[[]]

for jj in range(len(besttitle)):#len(title)-1
    
    thistitleindex=np.where(title==besttitle[jj])
    titlelist.append(thistitleindex[0][0])
    print(titlelist)
    Xtemp=X[:,titlelist]
    Xrealtesttemp=Xrealtest[:,titlelist]
    print(len(Xtemp[0]))
        
    perf=poolfit(TRAIN_TEST_SPLIT,EPOCH,CORE_NUM,Xtemp, y, paras,Xrealtesttemp)
    y_realtest_pred=perf[7]
    y_realtest_pred_reshaped = y_realtest_pred.reshape(1, -1)
    append_filename = Path('.', DIR, 'All_Realtestpred_' + c_time + '.txt')
    with open(append_filename, 'a+') as f:
        np.savetxt(f, y_realtest_pred_reshaped, fmt='%s', delimiter=',')
    
    
    y_realtest_pred_std=perf[8]
    y_realtest_pred_reshaped_std = y_realtest_pred_std.reshape(1, -1)
    append_filename = Path('.', DIR, 'All_Realtestpredstd_' + c_time + '.txt')
    with open(append_filename, 'a+') as f:
        np.savetxt(f, y_realtest_pred_reshaped_std, fmt='%s', delimiter=',')
    
        
    full_m=perf[3]
    full_m = np.array(full_m)
    test_idx_m = perf[4]
    test_idx_m = np.array(test_idx_m)
    save_name = 'XGBoost_02a_Test_Index_'+str(len(Xtemp[0]))+'_'+c_time+'.csv'
    save_name = Path('.', DIR, save_name)
    np.savetxt(save_name, test_idx_m, fmt='%d', delimiter=',')
    shap_m=perf[6]
    save_name = 'SHAP_Matrix_'+str(len(titlelist))+'_'+c_time+'.csv'
    save_name = Path('.', DIR, save_name)
    np.savetxt(save_name, shap_m, fmt='%s', delimiter=',')
    save_nameX = 'Feature_Matrix_'+str(len(titlelist))+'_'+c_time+'.csv'
    save_nameX = Path('.', DIR, save_nameX)
    np.savetxt(save_nameX, Xtemp, fmt='%s', delimiter=',')
    test_data_m = []
    for i in range(X.shape[0]):
        test_data_m.append([])
    for i in range(test_idx_m.shape[0]):
        for j in range(test_idx_m.shape[1]):
            test_data_m[test_idx_m[i, j]].append(full_m[i, test_idx_m[i, j]])
    test_upper_l = []
    test_lower_l = []
    test_mean_l = []
    test_median_l = []
    test_std_l = []
    for i in range(X.shape[0]):
        test_upper_l.append(max(test_data_m[i]))
        test_lower_l.append(min(test_data_m[i]))
        test_mean_l.append(np.mean(test_data_m[i]))
        test_median_l.append(np.median(test_data_m[i]))
        test_std_l.append(np.std(test_data_m[i]))
        
    true_y = y.flatten().tolist()
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_axes([0.11, 0.08, 0.88, 0.815])
    sc = ax.scatter(true_y, test_mean_l, alpha=0.55, c=test_std_l, cmap='viridis', marker='o')
    left_limit = min(min(true_y)-1, min(test_mean_l)-1)
    right_limit = max(max(true_y)+1, max(test_mean_l)+1)
    ax.plot([left_limit, right_limit], [left_limit, right_limit], color='#B22222', linestyle=':', linewidth = '2')
    ax.plot([left_limit, right_limit], [left_limit+1, right_limit+1], color='#FFA500', linestyle=':', linewidth = '2')
    ax.plot([left_limit, right_limit], [left_limit-1, right_limit-1], color='#FFA500', linestyle=':', linewidth = '2')
    ax.legend(['Correct', 'Correct+1', 'Correct-1', 'Mean of Test Prediction'], loc='upper left', fontsize=17, shadow=True)
    ax.set_xlabel('True Values', fontsize=17)
    ax.set_ylabel('Mean Values of Test Prediction', fontsize=17)
    plt.suptitle('Scatter of Mean Test Prediction vs True of '+str(len(Xtemp[0]))+' topfeatures\n'+
                 'Mean Test:  MSE: '+str(round(perf[0], 4))+
                 '  MAE: '+str(round(perf[1], 4))+
                 '  R^2: '+str(round(perf[2], 4)), fontsize=21)
    cb = plt.colorbar(sc)
    cb.set_label('Standard Deviation of Test Predictions', fontsize=17)
    plt.grid(which='major', color='#D5D5D5', alpha=0.5)
    save_name = 'XGBoost_02b_Mean_Test_Prediction_Distribution_'+str(len(Xtemp[0]))+'_'+c_time+'.png'
    save_name = Path('.', DIR, save_name)
    plt.savefig(save_name)
    plt.close()  # 关闭当前图形，以释放内存
    #plt.show()
    mselist.append(round(perf[0],3))
    xgbscatterdata= np.column_stack((true_y, test_mean_l, test_std_l))
    save_name1 = 'XGBoost_02c_Mean_Test_Prediction_Distribution_'+str(len(Xtemp[0]))+'_topfeatures'+'_MSE'+str(round(perf[0],3))+'_MSEstd'+str(round(perf[5],3))+'_MAE'+str(round(perf[1],3))+'_R^sq'+str(round(perf[2],3))+besttitle[jj]+'.txt'
    save_name2 = Path('.', DIR, save_name1)
    with open(save_name2,"w") as f:
        np.savetxt(save_name2,xgbscatterdata, fmt='%s', delimiter=',', comments='!')
    f.close()


# In[18]:


len(Xtemp)


# In[19]:


print(mselist)
print(mselist.index(min(mselist)))
print(title[bestfeatlist])
titlelist=[]
Xtemp=[]
for jj in range(len(bestfeatlist)):#len(title)-1
    thistitleindex=np.where(title==besttitle[jj])
    titlelist.append(thistitleindex[0][0])
    print(titlelist)
    Xtemp=X[:,titlelist]
# Create a scatter plot
for jj in range(len(bestfeatlist)):#len(title)-1
    fig = plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(Xtemp[:,jj], shap_m[:, jj])
    plt.xlabel('Xtemp')
    plt.ylabel('First column of SHAP Matrix')
    plt.title('SHAP Scatter plot of '+title[titlelist[jj]])
    save_name = 'XGBoost_02b_Mean_Test_Prediction_Distribution_'+str(title[titlelist[jj]])+'_'+c_time+'.png'
    save_name = Path('.', DIR, save_name)
    plt.savefig(save_name)
    plt.close()  # 关闭当前图形，以释放内存

