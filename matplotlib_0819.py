# -*- coding: utf-8 -*-
"""
@author: lvfanghua
"""

import matplotlib.pyplot as plt

#传入x,y，用plot画图
plt.plot([1,0,9],[4,5,6])
#展示图画
plt.show()

#折线图
x = range(1,8)
y = [17,17,18,15,11,11,13]
plt.plot(x,y,color = 'red',alpha = 0.5,
         linestyle = '--',linewidth = 3,marker = 'o')
plt.show()

# =============================================================================
# 基础属性设置：
# color：设置线的颜色
# alpha：设置线的透明度
# linestyle：设置线的样式
# linewidth：设置线的宽度
# marker：折点的样式
# =============================================================================


# =============================================================================
# figure方式
# =============================================================================

import random

x = range(2,26,2)
y = [random.randint(15, 30) for i in x]

#设置画布figure参数,figsize设置窗体大小，dpi制定图像分辨率
plt.figure(figsize = (20,8),dpi = 80)
plt.plot(x,y)
plt.savefig('G:\\开课吧\\0819/figure.jpg')

#plt.show()会释放figure资源，所以用savefig之前不能用show()

# =============================================================================
# 设置x轴和y轴的刻度
# =============================================================================
x = range(2,26,2)
y = [random.randint(15, 30) for i in x]

plt.figure(figsize=(20,8),dpi=80)
plt.xticks(x)
#plt.yticks(y)
plt.yticks(range(min(y),max(y)+1))
plt.plot(x,y)
plt.show()

# =============================================================================
# 设置轴刻度标签及显示中文
# xticks的参数一代表标签的位置，参数二是标签的内容
# =============================================================================
#构造x轴刻度标签
x = range(2,26,2)
y = [random.randint(15, 30) for i in x]
plt.figure(figsize=(20,8),dpi=80)

x_ticks_label = ["{}:00".format(i) for i in x]
#rotation = 45 让标签旋转45度
plt.xticks(x,x_ticks_label,rotation = 45)
#构造y轴刻度标签
y_ticks_label = ["{}℃".format(i) for i in range(min(y),max(y)+1)]
plt.yticks(range(min(y),max(y)+1),y_ticks_label)

#显示中文
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.serif'] = ['KaiTi']
plt.title('自定义刻度label')
plt.plot(x,y)
plt.show()

# =============================================================================
# 一图多线
# =============================================================================
import random
y1 = [1,0,1,1,2,4,3,4,4,5,6,5,4,3,3,1,1,1,1,1]
y2 = [1,0,3,1,2,2,3,4,3,2,1,2,1,1,1,1,1,1,1,1]

x = range(11,31)

fig = plt.figure(figsize = (20,8),dpi = 80)
plt.plot(x,y1,color = 'red',label = '自己')
plt.plot(x,y2,color = 'blue',label ='同事')

#设置刻度
xticks_label = ["{}岁".format(i) for i in x]
plt.xticks(x,xticks_label,rotation = 45)
#绘制网格，alpha为网格的透明度
plt.grid(alpha = 0.4)
#设置图例
plt.legend()
plt.show()

# =============================================================================
# =============================================================================
# # 一图多个坐标系子图
# # 通过add_subplot切分figure画布，
# # 然后对每个add_subplot进行独立操作，
# # 操作方法与切分之前的figure相同
# =============================================================================
# =============================================================================

# add_subplot方法----给figure新增子图 
import numpy as np 
import matplotlib.pyplot as plt 
x = np.arange(1, 100) 
#新建figure对象 
fig=plt.figure(figsize=(20,10),dpi=80) 
#新建子图1 
ax1=fig.add_subplot(2,2,1) 
ax1.plot(x, x) 
#新建子图2 
ax2=fig.add_subplot(2,2,2) 
ax2.plot(x, x ** 2) 
ax2.grid(color='r', linestyle='--', linewidth=1,alpha=0.3) 
#新建子图3 
ax3=fig.add_subplot(2,2,3) 
ax3.plot(x, np.log(x)) 
plt.show()

# =============================================================================
# 设置坐标轴范围
# =============================================================================
import matplotlib.pyplot as plt 
import numpy as np 
x= np.arange(-10,11,1) 
y = x**2 
plt.plot(x,y) 
# 可以调x轴的左右两边 
# plt.xlim([-5,5]) 
# 只调一边 
# plt.xlim(xmin=-4) 
# plt.xlim(xmax=4) 
plt.ylim(ymin=0) 
plt.xlim(xmin=0) 
plt.show()

# =============================================================================
# 改变坐标轴的默认显示方式
# =============================================================================

import matplotlib.pyplot as plt 
import numpy as np 
y = range(0,14,2) 
# x轴的位置 
x = [-3,-2,-1,0,1,2,3] 
# plt.figure(figsize=(20,8),dpi=80) 
# 获得当前图表的图像 
ax = plt.gca() 

# 显示中文
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.serif'] = ['KaiTi']
# 显示负号
plt.rcParams['axes.unicode_minus']=False

# 设置图型的包围线 
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none') 
ax.spines['bottom'].set_color('blue') 
ax.spines['left'].set_color('red') 
# 设置底边的移动范围，移动到y轴的0位置,'data':移动轴的位置到交叉轴的指定坐标 
ax.spines['bottom'].set_position(('data', 0)) 
ax.spines['left'].set_position(('data', 0)) 
plt.plot(x,y) 
plt.show()

# =============================================================================
# 绘制散点图
# =============================================================================
'''题干 3月份每天最高气温 
a = [11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22, 22,22,23] ''' 
from matplotlib import pyplot as plt 

y = [11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22, 22,22,23] 
x = range(1,32) 
# 设置图形大小 
plt.figure(figsize=(20,8),dpi=80) 
# 使用scatter绘制散点图 
plt.scatter(x,y,label= '3月份') 
# plt.plot() 
# 调整x轴的刻度 
_xticks_labels = ['3月{}日'.format(i) for i in x] 
plt.xticks(x[::3],_xticks_labels[::3],rotation=45) 
plt.xlabel('日期') 
plt.ylabel('温度') 
# 显示中文
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.serif'] = ['KaiTi']
# 图例 
plt.legend() 
plt.show()

# =============================================================================
# 绘制条形图
# =============================================================================
'''http://58921.com/alltime 
假设你获取到了2019年内地电影票房前20的电影（列表a)和电影票房数据（列表b)，请展示该数据 
a = ['流浪地球','疯狂的外星人','飞驰人生','大黄蜂','熊出没·原始时代','新喜剧之王'] 
b = ['38.13','19.85','14.89','11.36','6.47','5.93'] ''' 
from matplotlib import pyplot as plt 
a = ['流浪地球','疯狂的外星人','飞驰人生','大黄蜂','熊出没·原始时代','新喜剧之王'] 
b = ['38.13','19.85','14.89','11.36','6.47','5.93'] 
# b =[38.13,19.85,14.89,11.36,6.47,5.93] 
plt.figure(figsize=(20,8),dpi=80) 
# =============================================================================
# 纵向条形图 
# =============================================================================
rects = plt.bar(range(len(a)),[float(i) for i in b],width=0.3,color= ['r','g','b','r','g','b']) 
plt.xticks(range(len(a)),a) 
plt.yticks(range(0,41,5),range(0,41,5)) 
# 在条形图上加标注(水平居中) 
for rect in rects: 
    height = rect.get_height() 
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.3, str(height),ha="center") 
plt.show()

# =============================================================================
# 横向条形图
# =============================================================================
from matplotlib import pyplot as plt
a = ['流浪地球','疯狂的外星人','飞驰人生','大黄蜂','熊出没·原始时代','新喜剧之王'] 
b = [38.13,19.85,14.89,11.36,6.47,5.93] 
plt.figure(figsize=(20,8),dpi=80) 
# 绘制条形图的方法 
''' height=0.3 条形的宽度 ''' 
# 绘制横向的条形图 :barh
# plt.bar(y,x) 
rects = plt.barh(range(len(a)),b,height=0.5,color='r') 
plt.yticks(range(len(a)),a,rotation=45) 
# 在条形图上加标注(水平居中) 
for rect in rects: 
    # print(rect.get_x()) 
    width = rect.get_width() 
    plt.text(width, rect.get_y()+0.5/2, str(width),va="center") 

plt.show()

# =============================================================================
# 并列和罗列条形图
# =============================================================================
import matplotlib.pyplot as plt 
import numpy as np 
index = np.arange(4) 
BJ = [50,55,53,60] 
Sh = [44,66,55,41] 
# 并列 
plt.bar(index,BJ,width=0.3) 
plt.bar(index+0.3,Sh,width=0.3,color='green') 
plt.xticks(index+0.3/2,index) 
plt.show()
# 罗列 
plt.bar(index,BJ,width=0.3) 
plt.bar(index,Sh,bottom=BJ,width=0.3,color='green') 
plt.show()

# =============================================================================
# 直方图
# =============================================================================
''' 现有250部电影的时长，希望统计出这些电影时长的分布状态(比如时长为100分钟到120分钟电影的数量，出 现的频率)等信息，
你应该如何呈现这些数据？ ''' 
# 1）准备数据 
time = [131,98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138,
        131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 
        110, 116, 117, 110, 128, 128, 115, 99,136, 126, 134, 95, 
        138, 117, 111,78, 132, 124, 113, 150, 110, 117, 86, 95, 144, 
        105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123, 86, 
        101, 99, 136,123,117, 119, 105, 137, 123, 128, 125, 104, 109, 
        134, 125, 127,105, 120, 107, 129, 116,108, 132, 103, 136, 118, 
        102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125,138, 
        109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140, 83, 
        110, 102,123, 107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 
        109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 
        118, 112, 135,115, 146, 137, 116, 103, 144, 83, 123, 111, 110, 
        111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 
        111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 
        114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 
        133, 137, 92,121, 112, 146, 97, 137, 105, 98, 117, 112, 81, 97, 
        139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 
        101, 110,105, 129, 137, 112, 120, 113, 133, 112, 83, 94, 146, 133, 
        101,131, 116, 111, 84, 137, 115, 122, 106, 144, 109, 123, 116, 
        111,111, 133, 150] 

# 2）创建画布 
plt.figure(figsize=(20, 8), dpi=100) 
# 3）绘制直方图 
# 设置组距 
distance = 2 
# 计算组数 
group_num = int((max(time) - min(time)) / distance) 
# 绘制直方图 
plt.hist(time, bins=group_num) 
# 修改x轴刻度显示 
plt.xticks(range(min(time), max(time))[::2]) 
# 添加网格显示 
plt.grid(linestyle="--", alpha=0.5) 
# 添加x, y轴描述信息 
plt.xlabel("电影时长大小") 
plt.ylabel("电影的数据量") 
# 4）显示图像 
plt.show()

# =============================================================================
# 绘制饼图
# =============================================================================
import matplotlib.pyplot as plt

#各部分标签
label_list = ["第一部分","第二部分","第三部分"]
#各部分大小
size = [55,35,10]
#颜色
color = ["red","green","blue"]
#各部分突出值
explode = [0,0.05,0]


# =============================================================================
# 绘制饼图的参数及返回值：
# explode：设置各部分突出 
# label:设置各部分标签 
# labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径 
# autopct：设置圆里面文本 
# shadow：设置是否有阴影 
# startangle：起始角度，默认从0开始逆时针转 
# pctdistance：设置圆内文本距圆心距离 
# 返回值: 
# patches : matplotlib.patches.Wedge列表(扇形实例) 
# l_text：label matplotlib.text.Text列表(标签实例) 
# p_text：label matplotlib.text.Text列表(百分比标签实例) 
# =============================================================================
plt.figure(figsize=(10, 8), dpi=100) 
patches, l_text, p_text = plt.pie(size, 
                                  explode=explode, 
                                  colors=color, 
                                  labels=label_list, 
                                  labeldistance=1.1, 
                                  autopct="%1.1f%%", 
                                  shadow=False, 
                                  startangle=90, 
                                  pctdistance=0.6) 
for t in l_text: 
    print(dir(t)) 

for t in p_text:
    t.set_size(18)

for i in patches: 
    i.set_color('pink') 
    break

plt.legend() 
plt.show()



