# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:21:51 2020

@author: lvfanghu
"""

# =============================================================================
# SQL中一些常用操作，用pandas怎么实现
# =============================================================================

import pandas as pd

data=pd.read_csv()

#select某表指定字段数据，返回两行

search=data[['a','id','name','class']].head(2)

#查询满足class=2条件的数据，返回两行

search2=data[data['class']==2].head()

#多条件查询，与和或

search3 = data[(data['a'] > 2) & (data['class'] == 2)].head(2)

search4 = data[(data['a'] > 2) | (data['class'] == 2)].head(2)

#空值判断

#某字段为空的观测
search5 = data[data['name'].isna()]

#某字段为非空的观测
search6 = data[data['name'].notna()]

#升降序排序

search7 = data[(data['a' >= 6])].sort_values(by = 'class',ascending = False)

#更新修改满足指定条件的记录,class=2 且 a>2的记录，把class值修改为9

search8 = data.loc[(data['class'] == 2) & (data['a'] > 2),'class'] = 9

#分组统计
#按class分组统计量

search9 = data.groupby('class').size()

#分组统计'a'的最大值、'id'的均值

import numpy as np

search10 = data.groupby('class').agg({'a':np.max,'id':np.mean})

#删除满足指定条件的记录

search11 = data.drop(data[(data['class'] == 9) & (data['a'] > 2)].index)

#union 和 join

t1=t1.dropna(axis=0,how='all')  # dropna中na为缺失；axis 指轴，0是行，1是列;how 是删除条件：any 任意一个为na则删除整行/列,all 整行/列为na才删除
ttr1=t1.drop_duplicates([u'合同号']) #drop_duplicates指对数据进行去重处理，例如ttr14=t14.drop_duplicates([u'合同号']，'last/first')指对数据根据合同号进行去重，结果保留其中的第一或最后一份数据
tc1=pd.merge(ttr1,tt1,how='left',left_on=u'合同号',right_on=u'合同号') #merge()数据合并函数，此处指对tf和zc进行左连接数据合并，连接依据为’合同号‘






