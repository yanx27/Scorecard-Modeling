评分卡模型建模流程
====  
#
数据导入和建立 
-------
* 读入数据 
导入数据集 [application.csv](https://github.com/yanx27/ScoreCard-Model-based-on-Machine-Learning/blob/master/application.csv)
* 选择合适的建模样本 
* 数据集划分成训练集和测试集
#
第一步：数据预处理
-------
* 数据清洗
时间、类型等
* 格式转换 
* 缺失值填补
#
第二步：变量衍生
* 考虑申请额度与收入的占比
* 考虑earliest_cr_line到申请日期的跨度，以月份记

#
第三步：分箱
-------
* 采用ChiMerge,要求分箱完之后：<br>
（1）不超过5箱<br> 
（2）Bad Rate单调<br> 
（3）每箱同时包含好坏样本<br> 
（4）特殊值如－1，单独成一箱<br>

* 连续型变量可直接分箱<br> 
* 类别型变量：<br> 
（a）当取值较多时，先用bad rate编码，再用连续型分箱的方式进行分箱<br> 
（b）当取值较少时：<br> 
  >（b1）如果每种类别同时包含好坏样本，无需分箱<br> 
  >（b2）如果有类别只包含好坏样本的一种，需要合并<br> 
#
第四步：WOE编码、计算IV
-------
* WOE的公式：<br>
![](https://github.com/yanx27/ScoreCard-Model-based-on-Machine-Learning/blob/master/alg1.png)
![](https://github.com/yanx27/ScoreCard-Model-based-on-Machine-Learning/blob/master/alg2.png)
* WOE 的值越高，代表着该分组中客户是坏客户的风险越低。
#
第五步：单变量分析和多变量分析，均基于WOE编码后的值
-------
* 选择IV高于0.02的变量
* IIV值是用来衡量某个变量对好坏客户区分能力的一个指标，IV值公式如下：
![](https://github.com/yanx27/ScoreCard-Model-based-on-Machine-Learning/blob/master/alg3.png)      
* 比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
# 
第六步：逻辑回归模型（或其他机器学习算法）
-------
* 要求：<br>
（1）变量显著<br> 
（2）系数为负<br> 
#
第七步：评估
-------
* 利用ks和AUC等评估指标（亦可使用混淆矩阵）
* KS值越大，表示模型能够将正、负客户区分开的程度越大。 
* 通常来讲，KS>0.2即表示模型有较好的预测准确性。
* KS绘制方式与ROC曲线略有相同，都要计算TPR和FPR。但是TPR和FPR都要做纵轴，横轴为把样本分成多少份。 
* 步骤： 
（1）按照分类模型返回的概率降序排列 
（2）把0-1之间等分N份，等分点为阈值，计算TPR、FPR 
（3）对TPR、FPR描点画图即可

* KS值即为Max(TPR-FPR)
