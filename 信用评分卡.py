# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:51:07 2020

@author: lvfanghu
"""
import pandas as pd

path='D:/study/项目1信用评分卡/项目代码/项目1-互联网金融信用评分卡模型构建/2.金融信用评分卡模型建模/'

data=pd.read_csv(path+'cs-training.csv')

desc=data.describe()

desc.to_excel(path+'var_describe.xlsx')

h=data.head(10)
h.to_excel(path+'head.xlsx')

#数据预处理
#缺失值处理

# =============================================================================
# 用随机森林填充缺失值MonthlyIncome，
# 思想：把未缺失的数值型特征当做样本集，缺失的当做待预估值，用随机森林做预测的结果来填充
# =============================================================================

datanum=data.ix[:,[5,0,1,2,3,4,6,7,8,9]]

MonthlyIncome_nnull=datanum[datanum.MonthlyIncome.notnull()].as_matrix()

MonthlyIncome_null = datanum[datanum.MonthlyIncome.isnull()].as_matrix()


X = MonthlyIncome_nnull[:,1:]
#目标变量
y = MonthlyIncome_nnull[:,0]

from sklearn.ensemble import RandomForestRegressor

rfreg = RandomForestRegressor(random_state=0,
                              n_estimators=200,max_depth=3,n_jobs=-1)

# =============================================================================
# max_features：筛选特征的比例，这里特征量较少，使用默认选择所有特征
# random_state:随机种子，设定固定的值，重复运行时候不会出现多种结果，便于复现展示
# n_estimators：投票前建立子树的数量
# max_depth：最大树深
# n_jobs：告诉引擎有多少处理器是它可以使用，“-1”是所有。
# =============================================================================

rfreg.fit(X,y)

#用训练结果对缺失值进行预估
predict = rfreg.predict(MonthlyIncome_null[:,1:])

# 将预测的结果填充回原数据缺失值
data.loc[(data.MonthlyIncome.isnull()), 'MonthlyIncome'] = predict

desc1=data.describe()

#data.dropna(how='all')：整行缺失才会删除；data.dropna(thresh=2)：缺失值个数大于2，该行才会被删
#data.dropna():只要有某一特征有缺失就删除整行，只有少数几列有少量缺失值时才可这么处理
#单列看的时候，data.dropna(subset=[1，2])：删除指定列中包含缺失值的行
data=data.dropna(subset=['NumberOfDependents'])


#异常值处理
#剔除年龄为0的样本
data=data[data['age']>0]
data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]

data['SeriousDlqin2yrs']=1-data['SeriousDlqin2yrs']

data = data.reset_index(drop=True)

#画箱线图检测
import matplotlib.pyplot as plt

fig,axes = plt.subplots()
data.boxplot(column='NumberOfTime30-59DaysPastDueNotWorse',ax=axes)
# column参数表示要绘制成箱形图的数据，可以是一列或多列
# by参数表示分组依据
 
axes.set_ylabel('values of NumberOfTime30-59DaysPastDueNotWorse')

y = data['SeriousDlqin2yrs']
X = data.ix[:,1:]

#EDA
import mglearn
data1 = data[data['RevolvingUtilizationOfUnsecuredLines']<2]
data1 = data[data['DebtRatio']<2]
data1 = data[data['MonthlyIncome']<40000]
data1['MonthlyIncome'].describe()

#iris_dataframe=pd.DataFrame(X,columns=X.columns.values)
iris_dataframe=pd.DataFrame(data1['MonthlyIncome'],columns=['MonthlyIncome'])
grr = pd.plotting.scatter_matrix(iris_dataframe,marker='o',c = y,hist_kwds={'bins':20},cmap=mglearn.cm3)



#最优分箱法自动分箱：又叫监督离散化，使用递归划分，将连续变量分为分段。分布大体为正态分布才能用
import numpy as np
import scipy.stats as st

def partition(y,X,n=20):
    r = 0
    good=y.sum()
    bad=y.count()-good
    #当n=2时，二维空间上只有两个点，只要连续变量没有重复值，必要单调相关，所以此时r为1或-1或无限接近
    while np.abs(r) < 1-0.000001:
        d1 = pd.DataFrame({"X": X, "Y": y, "Bucket": pd.qcut(X, n,duplicates="drop")})
        # 后面报错You can drop duplicate edges by setting the 'duplicates' kwarg，所以回到这里补充duplicates参数
        # pandas中使用qcut()，边界易出现重复值，如果为了删除重复值设置 duplicates=‘drop’，则易出现于分片个数少于指定个数的问题
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = st.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    print("=" * 60)
    print(d4)
    return d1,d4
          
pt1,p1t = partition(y,X.ix[:,'RevolvingUtilizationOfUnsecuredLines'],15)
pt2,p2t = partition(y,X.ix[:,'age'],15)
pt3,p3t = partition(y,X.ix[:,'DebtRatio'],15)
pt4,p4t = partition(y,X.ix[:,'MonthlyIncome'],15)

#对于不满足分布要求的连续变量，也可手动设置分箱节点或者用等距分箱
#duplicates="drop"：允许重复区间；duplicates="raise"：不允许重复区间
cutx3 = pd.DataFrame({"X": X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'], 
                      [X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'].min()-0.1, 0, 1, 3, 5, X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'].max()],
                      duplicates="drop")})
    
cutx6 = pd.DataFrame({"X": X.ix[:,'NumberOfOpenCreditLinesAndLoans'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberOfOpenCreditLinesAndLoans'], 
                      [X.ix[:,'NumberOfOpenCreditLinesAndLoans'].min()-0.1,  1, 2, 3, 5, X.ix[:,'NumberOfOpenCreditLinesAndLoans'].max()],
                      duplicates="drop")})
    
cutx7 = pd.DataFrame({"X": X.ix[:,'NumberOfTimes90DaysLate'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberOfTimes90DaysLate'], 
                      [X.ix[:,'NumberOfTimes90DaysLate'].min()-0.1,  0, 1, 3, 5, X.ix[:,'NumberOfTimes90DaysLate'].max()],
                      duplicates="drop")})
    
cutx8 = pd.DataFrame({"X": X.ix[:,'NumberRealEstateLoansOrLines'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberRealEstateLoansOrLines'], 
                      [X.ix[:,'NumberRealEstateLoansOrLines'].min()-0.1, 0,1,2, 3, X.ix[:,'NumberRealEstateLoansOrLines'].max()],
                      duplicates="drop")})
    
cutx9 = pd.DataFrame({"X": X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'], 
                      [X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'].min()-0.1, 0, 1, 3, X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'].max()],
                      duplicates="drop")})
    
cutx10 = pd.DataFrame({"X": X.ix[:,'NumberOfDependents'], "Y": y, "Bucket": pd.cut(X.ix[:,'NumberOfDependents'], 
                      [X.ix[:,'NumberOfDependents'].min()-0.1, 0, 1, 2, 3, 5, X.ix[:,'NumberOfDependents'].max()],
                      duplicates="drop")})

#相关性分析    
import seaborn as sns
sns.set(style="darkgrid") #这是seaborn默认的风格
corr = X.corr()#计算各变量的相关性系数
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})#绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()
    
    
#IV值变量筛选
# =============================================================================
#IV值预测能力标准：
# < 0.02: unpredictive
# 0.02 to 0.1: weak
# 0.1 to 0.3: medium
# 0.3 to 0.5: strong
# > 0.5: suspicious
# =============================================================================

#带计算IV值和分箱节点值的自动分箱
def partition_iv(y,X,n=20):
    r = 0
    good=y.sum()
    bad=y.count()-good
    #当n=2时，二维空间上只有两个点，只要连续变量没有重复值，必要单调相关，所以此时r为1或-1或无限接近
    while np.abs(r) < 1-0.000001:
        d1 = pd.DataFrame({"X": X, "Y": y, "Bucket": pd.qcut(X, n,duplicates="drop")})
        # 后面报错You can drop duplicate edges by setting the 'duplicates' kwarg，所以回到这里补充duplicates参数
        # pandas中使用qcut()，边界易出现重复值，如果为了删除重复值设置 duplicates=‘drop’，则易出现于分片个数少于指定个数的问题
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = st.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    #woe
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    #iv
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    
    print("=" * 60)
    print(d4)
    #保存各分箱节点的值
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua=X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe=list(d4['woe'].round(3))
    return d1,d4,iv,cut,woe

#不符合分布的连续变量计算iv值函数
    
def partition_n(y,X,split_list):
    cut = split_list
    good = y.sum()
    bad = y.count() - good
    cut1 = pd.DataFrame({'X':X,'y':y,'bucket':pd.cut(X,split_list,duplicates='drop')})
    cut2 = cut1.groupby('bucket',as_index=True)
    cut3 = pd.DataFrame(cut2.X.min(),columns = ['min'])
    cut3['min'] = cut2.min().X
    cut3['max'] = cut2.max().X
    cut3['goodrate'] = cut2.mean().y
    #woe计算
    cut3['woe'] = np.log((cut3['goodrate']/(1-cut3['goodrate']))/(good/bad))
    cut3['sum'] = cut2.sum().y
    woe = list(cut3['woe'].round(3))
    #分箱好客户占总的好客户占比
    cut3['goodallrate'] = cut3['sum']/good
    #
    cut3['count'] = cut2.count().y
    cut3['badallrate'] = (cut3['count']-cut3['sum'])/bad
    cut4 = (cut3.sort_index(by = 'min')).reset_index(drop=True)
    
    #iv值计算
    iv = ((cut3['goodallrate']-cut3['badallrate'])*cut3['woe']).sum()
    return cut1,cut4,iv,cut,woe

#计算各变量iv值
    
d1,d4,ivx1,cutx1,woex1 = partition_iv(y,X.ix[:,'RevolvingUtilizationOfUnsecuredLines'],15)
d1,d4,ivx2,cutx2,woex2 = partition_iv(y,X.ix[:,'age'],15)
d1,d4,ivx4,cutx4,woex4 = partition_iv(y,X.ix[:,'DebtRatio'],15)
d1,d4,ivx5,cutx5,woex5 = partition_iv(y,X.ix[:,'MonthlyIncome'],15)

cut1,cut4,ivx3,cutx3,woex3 = partition_n(y,X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'],
                                  [X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'].min()-0.1, 0, 1, 3, 5, X.ix[:,'NumberOfTime30-59DaysPastDueNotWorse'].max()])

cut1,cut4,ivx6,cutx6,woex6 = partition_n(y,X.ix[:,'NumberOfOpenCreditLinesAndLoans'],
                                  [X.ix[:,'NumberOfOpenCreditLinesAndLoans'].min()-0.1,  1, 2, 3, 5, X.ix[:,'NumberOfOpenCreditLinesAndLoans'].max()])

cut1,cut4,ivx7,cutx7,woex7 = partition_n(y,X.ix[:,'NumberOfTimes90DaysLate'],
                                  [X.ix[:,'NumberOfTimes90DaysLate'].min()-0.1,  0, 1, 3, 5, X.ix[:,'NumberOfTimes90DaysLate'].max()])

cut1,cut4,ivx8,cutx8,woex8 = partition_n(y,X.ix[:,'NumberRealEstateLoansOrLines'],
                                  [X.ix[:,'NumberRealEstateLoansOrLines'].min()-0.1, 0,1,2, 3, X.ix[:,'NumberRealEstateLoansOrLines'].max()])

cut1,cut4,ivx9,cutx9,woex9 = partition_n(y,X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'],
                                  [X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'].min()-0.1, 0, 1, 3, X.ix[:,'NumberOfTime60-89DaysPastDueNotWorse'].max()])

cut1,cut4,ivx10,cutx10,woex10 = partition_n(y,X.ix[:,'NumberOfDependents'],
                                  [X.ix[:,'NumberOfDependents'].min()-0.1, 0, 1, 2, 3, 5, X.ix[:,'NumberOfDependents'].max()])

    

#生成IV图
ivlist=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]#各变量IV
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴的标签
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, ivlist, width=0.4)#生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()

#DebtRatio、MonthlyIncome、NumberOfOpenCreditLinesAndLoans、NumberRealEstateLoansOrLines和NumberOfDependents变量的IV值明显较低，可以删掉
X = X.drop(['DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)

#把原数据替换成woe
def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

datawoe = X
datawoe['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(replace_woe(X['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
datawoe['age'] = pd.Series(replace_woe(X['age'], cutx2, woex2))
datawoe['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(replace_woe(X['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
datawoe['NumberOfTimes90DaysLate'] = pd.Series(replace_woe(X['NumberOfTimes90DaysLate'], cutx7, woex7))
datawoe['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(replace_woe(X['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))



import datetime
#lr模型训练
#训练集切分目标和特征
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    datawoe, y, test_size=0.3, random_state=42)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


Y_model = y_train
#自变量
X_model = X_train

#设置multi_class='ovr'，会训练出“类别数”个分类器，构建样本时需要原始label即可
#基于sklearn.linear_model的lr模型训练
from sklearn.linear_model import LogisticRegression
btime = datetime.datetime.now() 
lr_clf = LogisticRegression(random_state=0, solver='sag',multi_class='ovr', verbose = 1)
lr_clf.fit(X_model, Y_model)

print ('all tasks done. total time used:%s s.\n\n'%((datetime.datetime.now() - btime).total_seconds()))

#模型评估
# 1、AUC
from sklearn.preprocessing import label_binarize
y_pred_pa = lr_clf.predict_proba(X_test)
y_test_oh = label_binarize(y_test, classes=[0,1])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

print ('调用函数auc：', roc_auc_score(y_test_oh, y_pred_pa[:,1], average='micro'))

#  2、混淆矩阵
y_pred = lr_clf.predict(X_test)
confusion_matrix(y_test, y_pred)

#  3、经典-精确率、召回率、F1分数
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report

precision_score(y_test, y_pred,average='micro')
recall_score(y_test, y_pred,average='micro')
f1_score(y_test, y_pred,average='micro')

# 4、模型报告
print(classification_report(y_test, y_pred , digits=4))

# 保存模型
from sklearn.externals import joblib
joblib.dump(lr_clf, 'C:\\Users\\lvfanghu\\Desktop\\0519/lr_clf.m')

#已有模型读取
clf_lr = joblib.load("C:\\Users\\lvfanghu\\Desktop\\0519/lr_clf.m")

clf_lr.predict(X_test)

print("模型权重系数："+str(clf_lr.coef_)+"常数项b："+str(clf_lr.intercept_))
coe=clf_lr.coef_

#基于statsmodels包的模型训练和变量显著性检验
import statsmodels.api as sm
X1=sm.add_constant(X_model)
logit=sm.Logit(Y_model,X1)
result=logit.fit()
print(result.summary())

#模型检验
from sklearn.metrics import roc_curve,auc
X3 = sm.add_constant(X_test)
resu = result.predict(X3)#进行预测
fpr, tpr, threshold = roc_curve(y_test, resu)
rocauc = auc(fpr, tpr)#计算AUC
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)#生成ROC曲线
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('真正率')
plt.xlabel('假正率')
plt.show()

#信用评分：把lr模型转化为标准评分卡形式





coe=list(coe[0])
# 我们取600分为基础分值，PDO为20（每高20分好坏比翻一倍），好坏比取20。
p = 20 / np.log(2)
q = 600 - 20 * np.log(20) / np.log(2)
baseScore = round(q + p * coe[0], 0)


#计算分数函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores
#计算各变量得分情况：
# 各项部分分数
x1 = get_score(coe[0], woex1, p)
x2 = get_score(coe[1], woex2, p)
x3 = get_score(coe[2], woex3, p)
x7 = get_score(coe[3], woex7, p)
x9 = get_score(coe[4], woex9, p)

#根据变量计算分数
def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

test1 = pd.concat([y_test, X_test], axis=1)

test1['BaseScore']=pd.Series(np.zeros(len(test1)))+baseScore
test1['x1'] = pd.Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
test1['x2'] = pd.Series(compute_score(test1['age'], cutx2, x2))
test1['x3'] = pd.Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
test1['x7'] = pd.Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7))
test1['x9'] = pd.Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))
test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] +test1['x9']  + baseScore
test1.to_csv('ScoreData.csv', index=False)
