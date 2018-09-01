# *_*coding:utf-8 *_*
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt

#读取EXCEL数据
fname="XX.xlsx"
bk=xlrd.open_workbook(fname)
bk = xlrd.open_workbook(fname)
shxrange = range(bk.nsheets)
try:
    sh = bk.sheet_by_name("Sheet2")
except:
    print("no sheet in %s named Sheet1" % fname)
nrows = sh.nrows
ncols = sh.ncols
# print("nrows %d, ncols %d" % (nrows, ncols))
row_list = []
for i in range(1, nrows):
    row_data = sh.row_values(i)
    row_list.append(row_data)
data=pd.DataFrame(row_list)
data.columns = ['cus_num', 'name', 'id', 'cell', 'scorebankv2','scoreconsoffv2', 'scorecreditbt', 'scorelargecashv1', 'scorelargecashv2', 'scorepettycashv1',"y"]
# print(data.head())

#KS计算
def ks(actuals,predictionScore,name=None,interval_num=None):
    if interval_num==None:
        num=10
    else:
        num=interval_num
    if len(actuals)!=len(predictionScore):
        print("invalid length between actuals and predictionScore")
    else:
        predictionScore=predictionScore.convert_objects(convert_numeric=True)
        if len(predictionScore[pd.isnull(predictionScore)]):
            null_true = pd.isnull(predictionScore)
            predictionScore = predictionScore[-null_true]
            actuals = actuals[-null_true]
        ksdata = pd.DataFrame()
        ksdata["label"] = actuals
        ksdata["prediction"] = predictionScore
        ksdata.columns = ['actuals', 'predictionScore']
        ksdata.sort_values("predictionScore", inplace=True)
        ksdata.index = np.arange(len(ksdata))
        index = []
        for i in range(num+1):
            index.append(int(i * len(ksdata) / num))
        index[len(index) - 1] = index[len(index) - 1] - 1
        FPR = []
        TPR = []
        KS = []
        for i in index:
            TP = len(ksdata[(ksdata["predictionScore"] < (ksdata["predictionScore"][i])) & (ksdata["actuals"] == 0)])
            FP = len(ksdata[(ksdata["predictionScore"] < (ksdata["predictionScore"][i])) & (ksdata["actuals"] == 1)])
            TN = len(ksdata[(ksdata["predictionScore"] >= (ksdata["predictionScore"][i])) & (ksdata["actuals"] == 1)])
            FN = len(ksdata[(ksdata["predictionScore"] >= (ksdata["predictionScore"][i])) & (ksdata["actuals"] == 0)])
            TPR.append(TP / (TP + FN))
            FPR.append(FP / (FP + TN))
            KS.append((FP / (FP + TN)) - (TP / (TP + FN)))
        t = np.argmax(np.array(KS))
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(num+1), TPR, c='green', linewidth=2.3, linestyle="-", label="TPR")
        plt.plot(np.arange(num+1), FPR, c='blue', linewidth=2.3, linestyle="-", label="FPR")
        plt.plot(np.arange(num+1), KS, c="m", linewidth=2.5, linestyle="-", label="KS")
        plt.xlabel('Rank', fontsize=16)
        plt.ylabel('Percentage Responders Captured', fontsize=16)
        if name==None:
            plt.title('KS Plot', fontsize=16)
        else:
            plt.title('KS Plot of %s ' % name, fontsize=16)
        # plt.text(.8, .6, 'ks=%f'%max(KS),fontsize=16)
        plt.annotate('ks=%f' % max(KS), fontsize=16, xy=(t - 0.1, .45), xytext=(.7, .6),
                     arrowprops=dict(facecolor='k', shrink=0.05, connectionstyle="arc3,rad=.1"))
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.plot([t, t], [TPR[t], FPR[t]], color='red', linewidth=2.5, linestyle="--")
        plt.show()
        print('ks=%f' % max(KS))
    return max(KS)

ks(actuals=data["y"],predictionScore=data["scorebankv2"],name="scorebankv2",interval_num=20)
ks(actuals=data["y"],predictionScore=data["scoreconsoffv2"],name="scoreconsoffv2")
ks(actuals=data["y"],predictionScore=data["scorecreditbt"],name="scorecreditbt",interval_num=30)
ks(actuals=data["y"],predictionScore=data["scorelargecashv1"],name="scorelargecashv1")
ks(actuals=data["y"],predictionScore=data["scorelargecashv2"],name="scorelargecashv2")
ks(actuals=data["y"],predictionScore=data["scorepettycashv1"],name="scorepettycashv1")
