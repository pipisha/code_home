# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:08:33 2020

@author: firstuser
"""
# =============================================================================
# read data
# =============================================================================
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import os 

path='F:/项目2与代码/data/PPD-First-Round-Data-Update/'
os.chdir(path)

f_train1 = pd.read_csv('Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gbk')
f_train2 = pd.read_csv('Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv',encoding='gbk')
f_train3 = pd.read_csv('Training Set/PPD_LogInfo_3_1_Training_Set.csv',encoding='gbk')
f_test1 = pd.read_csv('Test Set/PPD_Master_GBK_2_Test_Set.csv',encoding='gb18030')
f_test2 = pd.read_csv('Test Set/PPD_Userupdate_Info_2_Test_Set.csv',encoding='gbk')
f_test3 = pd.read_csv('Test Set/PPD_LogInfo_2_Test_Set.csv',encoding='gbk')

# 训练集和测试集合并
f_train1['sample_status'] = 'train'
f_test1['sample_status'] = 'test'
df1 = pd.concat([f_train1,f_test1],axis=0).reset_index(drop=True)
df2 = pd.concat([f_train2,f_test2],axis=0).reset_index(drop=True)
df3 = pd.concat([f_train3,f_test3],axis=0).reset_index(drop=True)

#df1.head()

# 保存数据至本地
df1.to_csv(path+'/data_input1.csv',encoding='gb18030',index=False)
df2.to_csv(path+'/data_input2.csv',encoding='gb18030',index=False)
df3.to_csv(path+'/data_input3.csv',encoding='gb18030',index=False)


# =============================================================================
# eda_and_data_clearning
# =============================================================================

import numpy as np 
import math 
import pandas as pd 
pd.set_option('display.float_format',lambda x:'%.3f' % x)
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns 
sns.set_palette('muted')
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import os
import score_card as sc
import missingno
#missingno是用于可视化缺失值的库
os.chdir(path)

# 导入 data_input 处理好的数据
df1 = pd.read_csv('data_input1.csv',encoding='gb18030')
df1.shape

# 样本的好坏比
df1.target.value_counts()

#借款成交量的时间趋势变化

# 借款成交时间的范围
import datetime as dt
df1['ListingInfo'] = pd.to_datetime(df1.ListingInfo)
# 每个月份的用户数分布
df1['month'] = df1.ListingInfo.dt.strftime('%Y-%m')
# 绘制成交量的时间趋势图
plt.figure(figsize=(10,4))
plt.title('借款成交量的时间趋势图')
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#按'month'分组count
sns.countplot(data=df1.sort_values('month'),x='month')
plt.show()

#违约情况的时间趋势分析

month_group = df1.groupby('month') # 根据月份计算每个月的违约率
time_bad_trend = pd.DataFrame()
time_bad_trend['total'] = month_group.target.count()
time_bad_trend['bad'] = month_group.target.sum()
time_bad_trend['bad_rate']=time_bad_trend['bad']/time_bad_trend['total']
time_bad_trend = time_bad_trend.reset_index()
time_bad_trend['bad_rate'].fillna(0,inplace=True)

plt.figure(figsize=(12,4))
plt.title('违约率的时间趋势图')
sns.pointplot(data=time_bad_trend,x='month',y='bad_rate',linestyles='-')
plt.show()

#数据清洗

# 检查数值型变量的缺失
# 原始数据中-1作为缺失的标识，将-1替换为np.nan
data1 = df1.drop(['ListingInfo','month'],axis=1)
data1 = data1.replace({-1:np.nan})

# 缺失变量的数据可视化
missing_df =sc.missing_cal(data1)
missing_col = list(missing_df[missing_df.missing_pct>0].col)
missingno.bar(data1.loc[:,missing_col])

# 删除缺失率在80%以上的变量
data1 = sc.missing_delete_var(df=data1,threshold=0.8)
data1.shape

# 样本的趋势个数可视化:横着算缺失个数，并排序展示
sc.plot_missing_user(df=data1,plt_size=(16,5))

# 删除变量缺失个数在100个以上的用户
data1 = sc.missing_delete_user(df=data1,threshold=100)

# 同值化处理：一个变量百分之90都是同样的值，相当于常量，对模型没意义
base_col = [x for x in data1.columns if x!='target']
data1 = sc.const_delete(col_list=base_col,df=data1,threshold=0.9)
data1 = data1.reset_index(drop=True)
data1.shape

# 保存数据至本地
data1.to_csv(path+'/data1_clean.csv',encoding='gb18030',index=False)


# =============================================================================
# preprocessing_and_feature_engineering
# =============================================================================

import numpy as np
import pandas as pd 
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os 
os.chdir(path)

#用户数据表

# 导入data_EDA_clean处理过的数据
data1 = pd.read_csv('data1_clean.csv',encoding='gbk')

# 导入变量明细表
var_info = pd.read_csv('var_info.csv',encoding='utf-8')

base_col = list(data1.columns)
var_info2 = var_info[var_info.变量名称.isin(base_col)].reset_index(drop=True)
var_info2.变量类型.value_counts() 

#类别型变量
# 类别型变量的描述性分析
cate_col = list(var_info2[var_info2.变量类型=='Categorical'].变量名称)
# 数值型类别变量的desc
data1.loc[:,cate_col].describe().T.assign(nuniq = data1.loc[:,cate_col].apply(lambda x:x.nunique()),
                                          missing_pct = data1.loc[:,cate_col].apply(lambda x:(len(x)-x.count())/len(x)))

# =============================================================================
# UserInfo_2,UserInfo_4,UserInfo_8，UserInfo_20: 城市信息, 需要做降基处理
# UserInfo_7，UserInfo_19：省份信息
# UserInfo_9：运营商类型,需要做清洗
# WeblogInfo_20：微博信息，需要做降基处理
# =============================================================================

# 先对所有字符型变量作去空格处理
for col in data1.select_dtypes(include='O').columns:
    data1[col] = data1[col].map(lambda x:str(x).strip())
    
#省份
# =============================================================================
# 原数据有两个省份字段，推测一个为用户的户籍地址，另一个为用户居住地址所在省份，由此可衍生的字段为：
# 1.省份二值化，通过违约率将单个省份衍生为二值化特征，分为户籍省份和居住地省份
# 2.户籍省份和居住地省份是否一致，推测不一致的用户大部分为外来打工群体，相对违约率会高一点
# ps: 计算违约率时要考虑该省份的借款人数，如果人数太少，参考价值不大
# =============================================================================

# 计算各省份违约率
def plot_pro_badrate(df,col):
    group = df.groupby(col)
    df = pd.DataFrame()
    df['total'] = group.target.count()
    df['bad'] = group.target.sum()
    df['badrate'] = df['bad']/df['total']
    # 筛选出违约率排名前5的省份
    print(df.sort_values('badrate',ascending=False).iloc[:5,:])
    
# 户籍地址
plot_pro_badrate(data1,'UserInfo_19')

# 西藏自治区的人数太少，不具有参考价值，剔除后再计算
plot_pro_badrate(data1[~(data1.UserInfo_19=='西藏自治区')],'UserInfo_19')

# 居住地址
plot_pro_badrate(data1,'UserInfo_7')

# 户籍省份的二值化衍生
data1['is_tianjin_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='天津市' else 0,axis=1)
data1['is_shandong_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='山东省' else 0,axis=1)
data1['is_jilin_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='吉林省' else 0,axis=1)
data1['is_sichuan_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='四川省' else 0,axis=1)
data1['is_heilongj_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='黑龙江省' else 0,axis=1)

# 居住地址省份的二值化衍生
data1['is_tianjin_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='天津' else 0,axis=1)
data1['is_shandong_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='山东' else 0,axis=1)
data1['is_sichuan_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='四川' else 0,axis=1)
data1['is_hunan_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='湖南' else 0,axis=1)
data1['is_jilin_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='吉林' else 0,axis=1)

# 户籍省份和居住地省份不一致衍生
data1.UserInfo_19.unique()

data1.UserInfo_7.unique()

# 将UserInfo_19改成和居住地址省份相同的格式
UserInfo_19_change = []
for i in data1.UserInfo_19:
    if i=='内蒙古自治区' or i=='黑龙江省':
        j = i[:3]
    else:
        j=i[:2]
    UserInfo_19_change.append(j)
    
is_same_province=[]
# 判断UserInfo_7和UserInfo_19是否一致
for i,j in zip(data1.UserInfo_7,UserInfo_19_change):
    if i==j:
        a = 1
    else:
        a = 0
    is_same_province.append(a)
    
data1['is_same_province'] = is_same_province

# 删除原有的变量
data1 = data1.drop(['UserInfo_19','UserInfo_7'],axis=1)
data1.shape

#运营商

# 将运营商信息转换为哑变量
data1 = data1.replace({'UserInfo_9':{'中国移动':'china_mobile',
                                     '中国电信':'china_telecom',
                                     '中国联通':'china_unicom',
                                     '不详':'operator_unknown'}})
#get_dummies:独热编码
oper_dummy = pd.get_dummies(data1.UserInfo_9)
data1 = pd.concat([data1,oper_dummy],axis=1)
# 删除原变量
data1 = data1.drop(['UserInfo_9'],axis=1)
data1.shape

#城市

# =============================================================================
# 原数据中有4个城市信息，推测为用户登录的IP地址城市，衍生的逻辑为：
# 1. 通过xgboost挑选比较重要的城市变量，进行二值化衍生
# 2. 由4个城市特征的非重复项计数可衍生成 登录IP地址的变更次数
# 3. 根据城市的一线/二线/三线进行降基处理，再转化为二值化特征
# =============================================================================

# 计算4个城市特征的非重复项计数，观察是否有数据异常
for col in ['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']:
    print('{}:{}'.format(col,data1[col].nunique()))
    print('\t')
    
# UserInfo_8相对其他特征nunique较大，发现有些城市有"市"，有些没有，需要做一下清洗
print(data1.UserInfo_8.unique()[:50])

# UserInfo_8清洗处理，处理后非重复项计数减小到400
data1['UserInfo_8']=[s[:-1] if s.find('市')>0 else s[:] for s in data1.UserInfo_8] 
data1.UserInfo_8.nunique()

# 根据xgboost变量重要性的输出吧对城市作二值化衍生
data1_temp1 = data1[['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20','target']]
area_list=[]
# 将四个城市变量都做亚编码处理
for col in data1_temp1:
    dummy_df = pd.get_dummies(data1_temp1[col])
    dummy_df = pd.concat([dummy_df,data1_temp1['target']],axis=1)
    area_list.append(dummy_df)

df_area1 = area_list[0]
df_area2 = area_list[1]
df_area3 = area_list[2]
df_area4 = area_list[3]

# 用xgboost建模
from xgboost.sklearn import XGBClassifier
x_area1 = df_area1[df_area1['target'].notnull()].drop(['target'],axis=1)
y_area1 = df_area1[df_area1['target'].notnull()]['target']
x_area2 = df_area2[df_area2['target'].notnull()].drop(['target'],axis=1)
y_area2 = df_area2[df_area2['target'].notnull()]['target']
x_area3 = df_area3[df_area3['target'].notnull()].drop(['target'],axis=1)
y_area3 = df_area3[df_area3['target'].notnull()]['target']
x_area4 = df_area4[df_area4['target'].notnull()].drop(['target'],axis=1)
y_area4 = df_area4[df_area4['target'].notnull()]['target']
xg_area1 = XGBClassifier(random_state=0).fit(x_area1,y_area1)
xg_area2 = XGBClassifier(random_state=0).fit(x_area2,y_area2)
xg_area3 = XGBClassifier(random_state=0).fit(x_area3,y_area3)
xg_area4 = XGBClassifier(random_state=0).fit(x_area4,y_area4)


# 输出变量的重要性
from xgboost import plot_importance
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
plot_importance(xg_area1,ax=ax1,max_num_features=10,height=0.4)
plot_importance(xg_area2,ax=ax2,max_num_features=10,height=0.4)
plot_importance(xg_area3,ax=ax3,max_num_features=10,height=0.4)
plot_importance(xg_area4,ax=ax4,max_num_features=10,height=0.4)

# 城市变量的二值化
data1['is_zibo_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='成都' else 0,axis=1)
data1['is_yantai_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='烟台' else 0,axis=1)

data1['is_zibo_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='成都' else 0,axis=1)
data1['is_weifang_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='潍坊' else 0,axis=1)

data1['is_zibo_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='成都' else 0,axis=1)
data1['is_shantou_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='汕头' else 0,axis=1)

data1['is_zibo_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='淄博市' else 0,axis=1)
data1['is_chengdu_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='成都市' else 0,axis=1)
data1['is_weifang_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='潍坊市' else 0,axis=1)

# 将四个城市变量改成同一的格式
data1['UserInfo_20'] = [i[:-1] if i.find('市')>0 else i[:] for i in data1.UserInfo_20]
# 城市变更次数变量衍生
city_df = data1[['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']]
city_change_cnt =[]
for i in range(city_df.shape[0]):
    a = list(city_df.iloc[i])
    city_count = len(set(a))
    city_change_cnt.append(city_count)
data1['city_change_cnt'] = city_change_cnt
# 删除原变量
data1 = data1.drop(['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20'],axis=1)
data1.shape

#微博
# 将字符型的nan转为众数
for col in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
    data1 = data1.replace({col:{'nan':np.nan}})
# 将缺失填充为众数
for col in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
    data1[col] = data1[col].fillna(data1[col].mode()[0])
    
# 微博变量的哑变量处理
data1['WeblogInfo_19'] = ['WeblogInfo_19_'+s for s in data1.WeblogInfo_19]
data1['WeblogInfo_21'] = ['WeblogInfo_21_'+s for s in data1.WeblogInfo_21]

for col in ['WeblogInfo_19','WeblogInfo_21']:
    dummy_df = pd.get_dummies(data1[col])
    data1 = pd.concat([data1,dummy_df],axis=1)
# 删除原变量
data1 = data1.drop(['WeblogInfo_19','WeblogInfo_21','WeblogInfo_20'],axis=1)
data1.shape

#数值型变量
# 数值型变量的缺失率分布
import missingno
num_col = list(var_info2[var_info2.变量类型=='Numerical'].变量名称)
missingno.bar(data1.loc[:,num_col])

# 数值型变量的描述性分析
num_desc = data1.loc[:,num_col].describe().T.assign(nuniq = data1.loc[:,num_col].apply(lambda x:x.nunique()),\
                                         misssing_pct  =data1.loc[:,num_col].apply(lambda x:(len(x)-x.count())/len(x)))\
                              .sort_values('nuniq')
num_desc.head(10)

#排序特征衍生
num_col2 = [col for col in num_col if col!='target']
# 筛选出只有数值型变量的数据集
num_data = data1.loc[:,num_col2]

# 排序特征衍生
for col in num_col2:
    num_data['rank'+col] = num_data[col].rank(method='max')/num_data.shape[0]

# 将排序特征转为单独的数据集
rank_col = [col for col in num_data.columns if col not in num_col2]
rank_df = num_data.loc[:,rank_col]

#periods变量衍生

# 生成只包含periods的临时表
periods_col = [i for i in num_col2 if i.find('Period')>0]
periods_col2 = periods_col+['target']
periods_data = data1.loc[:,periods_col2]

# 观察包含period1所有字段的数据，发现字段之间量级差异比较大，可能代表不同的含义，不适合做衍生
periods1_col = [col for col in periods_col if col.find('Period1')>0]
periods_data.loc[:,periods1_col].head()

# 观察后缀都为1的字段，发现字段数据的量级基本一致，可以对其做min,max,avg等统计值的衍生
period_1_col=[]
for i in range(0,102,17):
    col = periods_col[i]
    period_1_col.append(col)
periods_data.loc[:,period_1_col].head()

p_num_col=[]
# 将Period变量按照后缀数字存储成嵌套列表
for i in range(0,17,1):
    p_col=[]
    for j in range(i,102,17):
        col = periods_col[j]
        p_col.append(col)
    p_num_col.append(p_col)
    
# min,max,avg等统计值的衍生，并将衍生后的特征存成单独的数据集
periods_data = periods_data.fillna(0)
periods_fea_data=pd.DataFrame()
for j,p_list in zip(range(1,18,1),p_num_col):
    p_data = periods_data.loc[:,p_list]
    period_min=[]
    period_max=[]
    period_avg=[]
    for i in range(periods_data.shape[0]):
        a = p_data.iloc[i]
        period_min.append(np.min(a))
        period_max.append(np.max(a))
        period_avg.append(np.average(a))
    periods_fea_data['periods_'+str(j)+'_min'] = period_min
    periods_fea_data['periods_'+str(j)+'_max'] = period_max
    periods_fea_data['periods_'+str(j)+'_avg'] = period_avg
    
# 保存特征衍生后的数据集至本地
data1.to_csv(path+'/data1_process.csv',encoding='gb18030',index=False)
rank_df.to_csv(path+'/rank_feature.csv',encoding='gbk',index=False)
periods_fea_data.to_csv(path+'/periods_feature.csv',encoding='gbk',index=False)

#修改信息表
# =============================================================================
# 衍生的变量
# 1.最近的修改时间距离成交时间差
# 2.修改信息的总次数
# 3.每种信息修改的次数
# 4.按照日期修改的次数
# =============================================================================

df2 = pd.read_csv('data_input2.csv',encoding='gbk')

# 最近的修改时间距离成交时间差
# 时间格式的转换
df2['ListingInfo1'] = pd.to_datetime(df2['ListingInfo1'])
df2['UserupdateInfo2'] = pd.to_datetime(df2['UserupdateInfo2'])

# 计算时间差
time_span = df2.groupby('Idx',as_index=False).agg({'UserupdateInfo2':np.max,'ListingInfo1':np.max})
time_span['update_timespan'] = time_span['ListingInfo1']-time_span['UserupdateInfo2']
time_span['update_timespan'] = time_span['update_timespan'].map(lambda x:str(x))
time_span['update_timespan'] = time_span['update_timespan'].map(lambda x:int(x[:x.find('d')]))
time_span = time_span[['Idx','update_timespan']]

# 将UserupdateInfo1里的字符改为小写形式
df2['UserupdateInfo1'] = df2.UserupdateInfo1.map(lambda x:x.lower())
# 根据Idx计算UserupdateInfo2的非重复计数
group = df2.groupby(['Idx','UserupdateInfo1'],as_index=False).agg({'UserupdateInfo2':pd.Series.nunique})

# 每种信息修改的次数的衍生
user_df_list=[]
for idx in group.Idx.unique():
    user_df  = group[group.Idx==idx]
    change_cate = list(user_df.UserupdateInfo1)
    change_cnt = list(user_df.UserupdateInfo2)
    user_col  = ['Idx']+change_cate
    user_value = [user_df.iloc[0]['Idx']]+change_cnt
    user_df2 = pd.DataFrame(np.array(user_value).reshape(1,len(user_value)),columns=user_col)
    user_df_list.append(user_df2)
cate_change_df = pd.concat(user_df_list,axis=0)
cate_change_df.head()

# 将cate_change_df里的空值填为0
cate_change_df = cate_change_df.fillna(0)
cate_change_df.shape

# 修改信息的总次数，按照日期修改的次数的衍生
update_cnt = df2.groupby('Idx',as_index=False).agg({'UserupdateInfo2':pd.Series.nunique,
                                                         'ListingInfo1':pd.Series.count}).\
                      rename(columns={'UserupdateInfo2':'update_time_cnt',
                                      'ListingInfo1':'update_all_cnt'})
update_cnt.head()

# 将三个衍生特征的临时表进行关联
update_info = pd.merge(time_span,cate_change_df,on='Idx',how='left')
update_info = pd.merge(update_info,update_cnt,on='Idx',how='left')
update_info.head()

# 保存数据至本地
update_info.to_csv(path+'/update_feature.csv',encoding='gbk',index=False)


#登录信息表

# =============================================================================
# 衍生的变量
# 1.累计登录次数
# 2.登录时间的平均间隔
# 3.最近一次的登录时间距离成交时间差
# =============================================================================

df3 = pd.read_csv('data_input3.csv',encoding='gb18030')

# 累计登录次数
log_cnt = df3.groupby('Idx',as_index=False).LogInfo3.count().rename(columns={'LogInfo3':'log_cnt'})
log_cnt.head()

# 最近一次的登录时间距离当前时间差
df3['Listinginfo1']=pd.to_datetime(df3.Listinginfo1)
df3['LogInfo3'] = pd.to_datetime(df3.LogInfo3)
time_log_span = df3.groupby('Idx',as_index=False).agg({'Listinginfo1':np.max,
                                                       'LogInfo3':np.max})
time_log_span['log_timespan'] = time_log_span['Listinginfo1']-time_log_span['LogInfo3']
time_log_span['log_timespan'] = time_log_span['log_timespan'].map(lambda x:str(x))
time_log_span['log_timespan'] = time_log_span['log_timespan'].map(lambda x:int(x[:x.find('d')]))
time_log_span= time_log_span[['Idx','log_timespan']]
time_log_span.head()

# 登录时间的平均时间间隔
df4  = df3.sort_values(by=['Idx','LogInfo3'],ascending=['True','True'])

df4['LogInfo4'] = df4.groupby('Idx')['LogInfo3'].apply(lambda x:x.shift(1))

df4['time_span'] = df4['LogInfo3']-df4['LogInfo4']
df4['time_span'] = df4['time_span'].map(lambda x:str(x))
df4 = df4.replace({'time_span':{'NaT':'0 days 00:00:00'}})
df4['time_span'] = df4['time_span'].map(lambda x:int(x[:x.find('d')]))

avg_log_timespan = df4.groupby('Idx',as_index=False).time_span.mean().rename(columns={'time_span':'avg_log_timespan'})
avg_log_timespan.head()

log_info = pd.merge(log_cnt,time_log_span,how='left',on='Idx')
log_info = pd.merge(log_info,avg_log_timespan,how='left',on='Idx')
log_info.head()

log_info.to_csv(path+'/log_info_feature.csv',encoding='gbk',index=False)


# =============================================================================
# 特征选择
# =============================================================================

import numpy as np 
import math 
import pandas as pd 
pd.set_option('display.float_format',lambda x:'%.3f' % x)
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns 
sns.set_palette('muted')
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import os 
os.chdir(path)
import lightgbm as lgb 
from lightgbm import plot_importance

# 导入feature_processing处理过后的数据
data = pd.read_csv('data1_process.csv',encoding='gb18030')
periods_df = pd.read_csv('periods_feature.csv',encoding='gbk')
rank_df = pd.read_csv('rank_feature.csv',encoding='gbk')
update_info = pd.read_csv('update_feature.csv',encoding='gbk')
log_df = pd.read_csv('log_info_feature.csv',encoding='gbk')

# 合并衍生后的变量，data1不包含排序特征和periods衍生特征
data1 = pd.merge(data,update_info,on='Idx',how='left')
data1 = pd.merge(data1,log_df,on='Idx',how='left')
data1.shape

# data2包含排序特征和periods衍生特征
data2 = pd.concat([data1,rank_df,periods_df],axis=1)
data2.shape

data_idx = data.Idx
df1  =data1.drop(['Idx'],axis=1)# 删除Idx
# 测试集训练集的划分
train_fea = np.array(df1[df1['target'].notnull()][df1.sample_status=='train'].drop(['sample_status','target'],axis=1))
test_fea = np.array(df1[df1.sample_status=='test'].drop(['sample_status','target'],axis=1))
train_label = np.array(df1[df1['target'].notnull()][df1.sample_status=='train']['target']).reshape(-1,1)
test_label = np.array(df1[df1.sample_status=='test']['target']).reshape(-1,1)


fea_names = list(df1.drop(['sample_status','target'],axis=1).columns)# 特征名字存成列表
feature_importance_values = np.zeros(len(fea_names)) # 

# 训练10个lightgbm，并对10个模型输出的feature_importances_取平均
for _ in range(10):
    model = lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,n_jobs=-1,verbose = -1)
    model.fit(train_fea,train_label,eval_metric='auc',verbose = 30)
    #model.fit(train_fea,train_label,eval_metric='auc',eval_set = [(test_fea, test_label)],early_stopping_rounds=100,verbose = -1)
    feature_importance_values += model.feature_importances_/10

# 将feature_importance_values存成临时表
fea_imp_df1 = pd.DataFrame({'feature':fea_names,
                           'fea_importance':feature_importance_values})
fea_imp_df1 = fea_imp_df1.sort_values('fea_importance',ascending=False).reset_index(drop=True)
fea_imp_df1['norm_importance'] = fea_imp_df1['fea_importance']/fea_imp_df1['fea_importance'].sum() # 特征重要性value的归一化
fea_imp_df1['cum_importance'] = np.cumsum(fea_imp_df1['norm_importance'])# 特征重要性value的累加值
fea_imp_df1.head()

# 特征重要性可视化
plt.figure(figsize=(16,5))
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.subplot(1,2,1)
plt.title('特征重要性')
sns.barplot(data=fea_imp_df1.iloc[:10,:],x='norm_importance',y='feature')
plt.subplot(1,2,2)
plt.title('特征重要性累加图')
plt.xlabel('特征个数')
plt.ylabel('cum_importance')
plt.plot(list(range(1, len(fea_names)+1)),fea_imp_df1['cum_importance'], 'r-')
plt.show()

# 剔除特征重要性为0的变量
zero_imp_col = list(fea_imp_df1[fea_imp_df1.fea_importance==0].feature)
fea_imp_df11 = fea_imp_df1[~(fea_imp_df1.feature.isin(zero_imp_col))]
print('特征重要性为0的变量个数为 ：{}'.format(len(zero_imp_col)))
print(zero_imp_col)

# 剔除特征重要性比较弱的变量
low_imp_col = list(fea_imp_df11[fea_imp_df11.cum_importance>=0.99].feature)
print('特征重要性比较弱的变量个数为：{}'.format(len(low_imp_col)))
print(low_imp_col)

# 删除特征重要性为0和比较弱的特征
drop_imp_col = zero_imp_col+low_imp_col
mydf1 = df1.drop(drop_imp_col,axis=1)
mydf1.shape

# 加上训练集测试集状态，保存数据
sample_status = list(df1.sample_status)
mydf1['sample_status'] = sample_status
mydf1['Idx'] = data_idx
mydf1.to_csv(path+'/feature_select_data1.csv',encoding='gb18030',index=False)


# =============================================================================
# lightgbm model
# =============================================================================

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
plt.style.use('ggplot')
import seaborn as sns 
import score_card as sc
import warnings 
warnings.filterwarnings('ignore')

import lightgbm as lgb 
from lightgbm import plot_importance 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


import os 
os.chdir(path)

df  = pd.read_csv('feature_select_data1.csv',encoding='gb18030')
df.head()

# 基于skearn默认参数模型
df_train, df_test = train_test_split(df[df.sample_status=='train'])
x_train = df_train.drop(['Idx','sample_status','target'],axis=1)
x_test = df_test.drop(['Idx','sample_status','target'],axis=1)
y_train = df_train['target']
y_test = df_test['target']

import time
start = time.time()
lgb_sklearn = lgb.LGBMClassifier(random_state=0).fit(x_train,y_train)
end = time.time()
print('运行时间为{}秒'.format(round(end-start,0)))

# 默认参数模型的AUC
lgb_sklearn_pre = lgb_sklearn.predict_proba(x_test)[:,1]
sc.plot_roc(y_test,lgb_sklearn_pre)


# 原生的lightgbm
lgb_train = lgb.Dataset(x_train,y_train)
lgb_test = lgb.Dataset(x_test,y_test,reference=lgb_train)
lgb_origi_params = {'boosting_type':'gbdt',
              'max_depth':-1,
              'num_leaves':31,
              'bagging_fraction':0.8,
              'feature_fraction':0.8,
              'learning_rate':0.03,
              'metric': 'auc'}
start = time.time()
lgb_origi = lgb.train(train_set=lgb_train,
                      early_stopping_rounds=10,
                      num_boost_round=3000,
                      params=lgb_origi_params,
                      valid_sets=lgb_test)
end = time.time()
print('运行时间为{}秒'.format(round(end-start,0)))

# 原生的lightgbm的AUC
lgb_origi_pre = lgb_origi.predict(x_test)
sc.plot_roc(y_test,lgb_origi_pre)

#调参
# 确定最大迭代次数，学习率设为0.1 
base_parmas={'boosting_type':'gbdt',
             'learning_rate':0.03,
             'num_leaves':40,
             'max_depth':-1,
             'bagging_fraction':0.8,
             'feature_fraction':0.8,
             'lambda_l1':0,
             'lambda_l2':0,
             'min_data_in_leaf':20,
             'min_sum_hessian_inleaf':0.001,
             'metric':'auc'}
cv_result = lgb.cv(train_set=lgb_train,
                   num_boost_round=200,
                   early_stopping_rounds=5,
                   nfold=5,
                   stratified=True,
                   shuffle=True,
                   params=base_parmas,
                   metrics='auc',
                   seed=0)

print('最大的迭代次数: {}'.format(len(cv_result['auc-mean'])))
print('交叉验证的AUC: {}'.format(max(cv_result['auc-mean'])))

# num_leaves ，步长设为5
param_find1 = {'num_leaves':range(30,100,10)}
cv_fold = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
start = time.time()
grid_search1 = GridSearchCV(estimator=lgb.LGBMClassifier(learning_rate=0.1,
                                                         n_estimators = 51,
                                                         max_depth=-1,
                                                         min_child_weight=0.001,
                                                         min_child_samples=20,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0),
                             cv = cv_fold,
                             n_jobs=-1,
                             param_grid = param_find1,
                             scoring='roc_auc')
grid_search1.fit(x_train,y_train)
end = time.time()
print('运行时间为:{}'.format(round(end-start,0)))

print(grid_search1.best_score_)
print('\t')
print(grid_search1.best_params_)
print('\t')
print(grid_search1.best_score_)

# num_leaves,步长设为2 
param_find2 = {'num_leaves':range(16,64,16)}
grid_search2 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_weight=0.001,
                                                         min_child_samples=20,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            n_jobs=-1,
                            scoring='roc_auc',
                            param_grid = param_find2)
grid_search2.fit(x_train,y_train)
print(grid_search2.best_score_)
print('\t')
print(grid_search2.best_params_)
print('\t')
print(grid_search2.best_score_)

# 确定num_leaves 为30 ，下面进行min_child_samples 和 min_child_weight的调参，设定步长为5
param_find3 = {'min_child_samples':range(15,35,5),
               'min_child_weight':[x/1000 for x in range(1,4,1)]}
grid_search3 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         num_leaves=30,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find3,
                            n_jobs=-1)
start = time.time()
grid_search3.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search3.grid_scores_)
print('\t')
print(grid_search3.best_params_)
print('\t')
print(grid_search3.best_score_)

# 确定min_child_weight为0.001，min_child_samples为20,下面对subsample和colsample_bytree进行调参
param_find4 = {'subsample':[x/10 for x in range(5,11,1)],
               'colsample_bytree':[x/10 for x in range(5,11,1)]}
grid_search4 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_samples=20,
                                                         min_child_weight=0.001,
                                                         num_leaves=30,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find4,
                            n_jobs=-1)
start = time.time()
grid_search4.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search4.grid_scores_)
print('\t')
print(grid_search4.best_params_)
print('\t')
print(grid_search4.best_score_)

param_find5 = {'reg_lambda':[0.001,0.01,0.03,0.08,0.1,0.3],
               'reg_alpha':[0.001,0.01,0.03,0.08,0.1,0.3]}
grid_search5 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_samples=20,
                                                         min_child_weight=0.001,
                                                         num_leaves=30,
                                                         subsample=0.5,
                                                         colsample_bytree=0.6,
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find5,
                            n_jobs=-1)
start = time.time()
grid_search5.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search5.grid_scores_)
print('\t')
print(grid_search5.best_params_)
print('\t')
print(grid_search5.best_score_)

# 将最佳参数再次带入cv函数，设定学习率为0.005
best_params = {
    'boosting_type':'gbdt',
    'learning_rate':0.005,
    'num_leaves':30,
    'max_depth':-1,
    'bagging_fraction':0.5,
    'feature_fraction':0.6,
    'min_data_in_leaf':20,
    'min_sum_hessian_in_leaf':0.001,
    'lambda_l1':0.3,
    'lambda_l2':0.03,
    'metric':'auc'
}

best_cv = lgb.cv(train_set=lgb_train,
                 early_stopping_rounds=5,
                 num_boost_round=2000,
                 nfold=5,
                 params=best_params,
                 metrics='auc',
                 stratified=True,
                 shuffle=True,
                 seed=0)

print('最佳参数的迭代次数: {}'.format(len(best_cv['auc-mean'])))
print('交叉验证的AUC: {}'.format(max(best_cv['auc-mean'])))

lgb_single_model = lgb.LGBMClassifier(n_estimators=900,
                                learning_rate=0.005,
                                min_child_weight=0.001,
                                min_child_samples = 20,
                                subsample=0.5,
                                colsample_bytree=0.6,
                                num_leaves=30,
                                max_depth=-1,
                                reg_lambda=0.03,
                                reg_alpha=0.3,
                                random_state=0)
lgb_single_model.fit(x_train,y_train)

pre = lgb_single_model.predict_proba(x_test)[:,1]
print('lightgbm单模型的AUC：{}'.format(metrics.roc_auc_score(y_test,pre)))
sc.plot_roc(y_test,pre)


# =============================================================================
# 模型融合
# =============================================================================

import numpy as np 
import pandas as pd
import lightgbm as lgb
import random
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import os 
os.chdir(path)
import score_card as sc

# Master数据
df1 = pd.read_csv('feature_select_data1.csv',encoding='gb18030')
df1.shape

# 排序特征数据
rank_df = pd.read_csv('rank_feature.csv',encoding='gbk')
rank_df.shape

# periods衍生特征数据
periods_df = pd.read_csv('periods_feature.csv',encoding='gbk')
periods_df.shape

# 原生特征（不含排序特征和periods衍生特征）
feature1 = list(df1.columns)
# 排序特征和periods衍生特征
feature2 = list(rank_df.columns)+list(periods_df.columns)

# 对feature2进行随机打乱顺序
random.shuffle(feature2)

# 合并数据集
df = pd.concat([df1,rank_df,periods_df],axis=1)
df.shape

# 保存用户id
data_idx = df.Idx

# 定义lightgbm的bagging函数
def bagging_lightgbm(feature_fraction,bagging_fraction,ramdom_seed,n_feature):
    
    select_fea = feature1+feature2[:n_feature]
    
    data = df.loc[:,select_fea]
    train_x = data[data.sample_status=='train'].drop(['sample_status','target','Idx'],axis=1)
    train_y = data[data.sample_status=='train']['target']
    test_x = data[data.sample_status=='test'].drop(['sample_status','target','Idx'],axis=1)
    test_y = data[data.sample_status=='test']['target']
    
    test_user_id = list(data[data.sample_status=='test']['Idx'])
    
    
    dtrain = lgb.Dataset(train_x,train_y)
    dtest = lgb.Dataset(test_x,test_y)
    
    params={
        'boosting_type':'gbdt',
        'metric':'auc',
        'num_leaves':30,
        'min_data_in_leaf':20,
        'min_sum_hessian_in_leaf':0.001,
        'bagging_fraction':bagging_fraction,
        'feature_fraction':feature_fraction,
        'learning_rate':0.005,
    }
    
    #  寻找最佳的迭代次数
    cv_result = lgb.cv(train_set=dtrain,
                       early_stopping_rounds=10,
                       num_boost_round=1000,
                       nfold=5,
                       metrics='auc',
                       seed=0,
                       params=params,
                       stratified=True,
                       shuffle=True)
    max_auc = max(cv_result['auc-mean'])
    num_round = len(cv_result['auc-mean'])
    
    model = lgb.train(train_set=dtrain,early_stopping_rounds=10,num_boost_round=num_round,valid_sets=dtest,params=params)
    
    model_pre = list(model.predict(test_x))
    result_df = pd.DataFrame({'Idx':test_user_id,
                              'score':model_pre})
    return result_df

# 对随机种子，bagging_fraction，feature_fraction及特征数量进行随机扰动
random_seed = list(range(2018))
bagging_fraction = [i/1000.0 for i in range(500,1000)]
feature_fraction = [i/1000.0 for i in range(500,1000)]
n_feature = list(range(50,174,2))

random.shuffle(random_seed)
random.shuffle(bagging_fraction)
random.shuffle(feature_fraction)
random.shuffle(n_feature)

import time 
a=  time.time()
result_df_list=[]
# 建立30个子模型，保存各个子模型输出的结果
for i in range(5):
    result_df = bagging_lightgbm(feature_fraction=feature_fraction[i],
                                 n_feature=n_feature[i],
                                 ramdom_seed=random_seed[i],
                                 bagging_fraction=bagging_fraction[i])
    result_df_list.append(result_df)
    print(i)
# 对30个子模型的结果average，得到bagging模型的最终结果
prep_list = [list(x['prep']) for x in result_df_list]
bagging_prep= list(np.sum(score_list,axis=0)/30)
b = time.time()
print('运行时间:{}'.format(round(b-a,0)))

# bagging模型的AUC
test_y = list(df[df.sample_status=='test']['target'])
sc.plot_roc(y_label=test_y,y_pred=ss)


