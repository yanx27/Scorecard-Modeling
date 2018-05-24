import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pylab as plt
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model.logistic import LogisticRegression
def CareerYear(x):
    #对工作年限进行转换
    x = str(x)
    if x.find('nan') > -1:
        return -1
    elif x.find("10+")>-1:   #将"10＋years"转换成 11
        return 11
    elif x.find('< 1') > -1:  #将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))   #其余数据，去掉"years"并转换成整数


def DescExisting(x):
    #将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
        #time.mktime 不能读取1970年之前的日期
    else:
        yr = int(x[4:6])
        if yr <=17:
            yr = 2000+yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr,mth,1)


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x



'''
第一步：数据准备
'''
folderOfData = foldOfData = '/Users/imxly/Desktop/apply/A_Card/'
allData = pd.read_csv(folderOfData + 'application.csv',header = 0, encoding = 'latin1')
allData['term'] = allData['term'].apply(lambda x: int(x.replace(' months','')))
# 处理标签：Fully Paid是正常用户；Charged Off是违约用户
allData['y'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))

'''
由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
'''
allData1 = allData.loc[allData.term == 36]
trainData, testData = train_test_split(allData1,test_size=0.4,random_state=0)



'''
第二步：数据预处理
'''
# 将带％的百分比变为浮点数
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)
# 将工作年限进行转化，否则影响排序
trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)
# 将desc的缺失作为一种状态，非缺失作为另一种状态
trainData['desc_clean'] = trainData['desc'].map(DescExisting)
# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))
# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))
trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

'''
第三步：变量衍生
'''
# 考虑申请额度与收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)
# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)


'''
对于类别型变量，需要onehot（独热）编码，再训练GBDT模型
'''
num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc','limit_income','earliest_cr_to_app']
cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']

v = DictVectorizer(sparse=False)
X1 = v.fit_transform(trainData[cat_features].to_dict('records'))
#将独热编码和数值型变量放在一起进行模型训练
X2 = np.matrix(trainData[num_features])
X = np.hstack([X1,X2])
y = trainData['y']
# 未经调参进行GBDT模型训练
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)



y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1].T
print ("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred)) # 0.8919
print ("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob)) # 0.744



#%%
'''
第四步：在测试集上测试模型的性能
'''

def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)

# 将带％的百分比变为浮点数
testData['int_rate_clean'] = testData['int_rate'].map(lambda x: float(x.replace('%',''))/100)
# 将工作年限进行转化，否则影响排序
testData['emp_length_clean'] = testData['emp_length'].map(CareerYear)
# 将desc的缺失作为一种状态，非缺失作为另一种状态
testData['desc_clean'] = testData['desc'].map(DescExisting)
# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
testData['app_date_clean'] = testData['issue_d'].map(lambda x: ConvertDateStr(x))
testData['earliest_cr_line_clean'] = testData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))
# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
testData['mths_since_last_delinq_clean'] = testData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
testData['mths_since_last_record_clean'] = testData['mths_since_last_record'].map(lambda x:MakeupMissing(x))
testData['pub_rec_bankruptcies_clean'] = testData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))


# 考虑申请额度与收入的占比
testData['limit_income'] = testData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)
# 考虑earliest_cr_line到申请日期的跨度，以月份记
testData['earliest_cr_to_app'] = testData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)

#用训练集里的onehot编码方式进行编码
X1_test = v.transform(testData[cat_features].to_dict('records'))
X2_test = np.matrix(testData[num_features])
X_test = np.hstack([X1_test,X2_test])
y_test = np.matrix(testData['y']).T

#在测试集上测试GBDT性能
y_pred = gbm0.predict(X_test)
y_predprob = gbm0.predict_proba(X_test)[:,1].T
testData['predprob'] = list(y_predprob)
print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_predprob))
print ("KS is :{}".format(KS(testData, 'predprob', 'y')))
'''
Accuracy : 0.8881
AUC Score (Test): 0.676696
KS is :0.2595474280765302
'''




#%%
'''
GBDT调参
'''
# 1, 选择较小的步长(learning rate)后，对迭代次数(n_estimators)进行调参

X = pd.DataFrame(X)

param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=30,
                                  min_samples_leaf=5,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
best_n_estimator = gsearch1.best_params_['n_estimators']


# 2, 对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
param_test2 = {'max_depth':range(3,16), 'min_samples_split':range(2,10)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_
best_max_depth = gsearch2.best_params_['max_depth']

#3, 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split':range(10,101,10), 'min_samples_leaf':range(5,51,5)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                     max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_
best_min_samples_split, best_min_samples_leaf = gsearch3.best_params_['min_samples_split'],gsearch3.best_params_['min_samples_leaf']

#4, 对最大特征数max_features进行网格搜索
param_test4 = {'max_features':range(5,int(np.sqrt(X.shape[0])),5)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth, min_samples_leaf =best_min_samples_leaf,
               min_samples_split =best_min_samples_split, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_
best_max_features = gsearch4.best_params_['max_features']

#5, 对采样比例进行网格搜索
param_test5 = {'subsample':[0.6+i*0.05 for i in range(8)]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                                               min_samples_leaf =best_min_samples_leaf, max_features=best_max_features,random_state=10),
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X,y)
gsearch5.best_params_, gsearch5.best_score_
best_subsample = gsearch5.best_params_['subsample']


gbm_best = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                      min_samples_leaf =best_min_samples_leaf, max_features=best_max_features,subsample =best_subsample, random_state=10)
gbm_best.fit(X,y)


#在测试集上测试并计算性能
y_pred = gbm_best.predict(X_test)
y_predprob = gbm_best.predict_proba(X_test)[:,1].T
testData['predprob'] = list(y_predprob)
print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_predprob))
print ("KS is :{}".format(KS(testData, 'predprob', 'y')))

'''
Accuracy : 0.888
AUC Score (Test): 0.677570
KS is :0.25473
'''
#%%

#GBDT模型生成特征

# 利用GBDT的结果衍生新的特征
new_feature= gbm_best.apply(X)[:,:,0]
grd_enc = OneHotEncoder()
grd_enc.fit(new_feature)
x = grd_enc.transform(new_feature)
classifier=LogisticRegression()
classifier.fit(x,y)

new_feature_test= gbm_best.apply(X_test)[:,:,0]
x_test = grd_enc.transform(new_feature_test)
y_pred_lr = classifier.predict_proba(x_test)[:,1]
lr_pred = pd.DataFrame({'predprob':y_pred_lr, 'y': np.array(y_test)[:,0]})

print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_pred_lr))
print ("KS is :%f"%KS(lr_pred, 'predprob', 'y'))