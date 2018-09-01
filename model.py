# *_*coding:utf-8 *_*
import pandas as pd
import os
import numpy as np
import scorecard_function
import matplotlib.pyplot as plt
import xgboost

from sklearn.metrics import accuracy_score
from xgboost import plot_importance
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

os.chdir("E:\...")
print("data loading....")
train = pd.read_excel("....")
col = train.columns.tolist()[1:]

#建模
print("modeling....")
Y = train.flagy
X = train[col]
# X = X.drop(zero_importance_features,axis=1)
print("the shape of training data is :",X.shape)
#数据分块
N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=0)
#利用Lightgbm建模
print("modeling in LightGBM")
i=0
xx_cv = []
xx_cv2 = []
xx_pre = []
xx_ks = []
xx_ks2 = []
xx_accuracy = []
xx_best_iteration = []
xx_precision = []
for train_in,test_in in skf.split(X,Y):
    X_train,X_test,y_train,y_test = X.ix[train_in,:],X.ix[test_in,:],Y[train_in],Y[test_in]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 20,
        'max_bin':10,
        'learning_rate': 0.005,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'min_sum_hessian_in_leaf':8,
        'max_depth': 10,
        'bagging_freq': 9,
        'verbose': 0
    }
    i += 1
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=500,
                    early_stopping_rounds=200)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred2 = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    result1 = pd.DataFrame({"y": y_test, "prob": y_pred})
    result2 = pd.DataFrame({"y": y_train, "prob": y_pred2})
    ks = scorecard_function.KS(result1, 'prob', 'y')
    ks2 = scorecard_function.KS(result2, 'prob', 'y')
    print('test ks:%f'%ks)
    print('train ks:%f'%ks2)
    auc = roc_auc_score(result1['y'], result1['prob'])
    auc2 = roc_auc_score(result2['y'], result2['prob'])
    predictions = [round(value) for value in y_pred]
    accuracy = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    # 变量重要性可视化
    # ax = lgb.plot_importance(gbm, max_num_features=30)
    # plt.show()
    xx_ks.append(ks)
    xx_ks2.append(ks2)
    xx_cv.append(auc)
    xx_cv2.append(auc2)
    xx_accuracy.append(accuracy*100)
    xx_precision.append(precision*100)
    xx_best_iteration.append(gbm.best_iteration)
    xx_pre.append(gbm.predict(X_test, num_iteration=gbm.best_iteration))
print(xx_ks)
print(xx_ks2)
print('LightGBM:test ks {}, auc {}'.format(np.mean(xx_ks), np.mean(xx_cv)))
print('LightGBM:train ks {}, auc {}'.format(np.mean(xx_ks2), np.mean(xx_cv2)))
print("Mean of recall: %.2f%%" % (np.mean(xx_accuracy) ))
print("Mean of precision: %.2f%%" % (np.mean(xx_precision) ))
print("Mean of accuracy: %.2f%%" % (np.mean(xx_accuracy) ))
print("Best iteration:" , round(np.mean(xx_best_iteration)))

# 构建 Xgboost 模型
print("modeling in XGBoost....")
model = xgboost.XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.03, # 如同学习率
min_child_weight=1,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=8, # 构建树的深度，越大越容易过拟合
gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=0.8, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样
reg_lambda=2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
#scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
objective= 'binary:logistic', #多分类的问题 指定学习任务和相应的学习目标
#num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=200, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
 )
eval_set = [(X_test, y_test)] #每次加入模型后测试
model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc',
eval_set=eval_set, verbose=True)
# early_stopping_rounds ：多少次训练模型没有提升就结束模型
#eval_metric ：评估标准
#eval_set ：测试集
#verbose ：输出每次加入模型的效果
y_pred = model.predict_proba(X_test)
print('Start predicting...')
prob = [value[1] for value in y_pred]
result=pd.DataFrame({"y":y_test,"prob":prob})
ks = scorecard_function.KS(result, 'prob', 'y')
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(result['y'], result['prob'])
print('XGBoost:ks {}, auc {}'.format(ks, auc))
predictions = [round(value[1]) for value in y_pred]
# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#变量重要性
plot_importance(model,max_num_features=30)
plt.show()


