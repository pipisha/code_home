# -*- coding: utf-8 -*-
"""

@author: lvfanghu
"""

# =============================================================================
# 基于SVM支持向量机的人脸识别
# =============================================================================

# =============================================================================
#用支持向量机实现人脸的分类识别. 对输入的人脸图像, 使用PCA(主成分分析)将图像进行了降维处理, 然后将降维后的向量作为支持向量机的输入. 
# PCA降维的目的可以看作是特征提取, 将图像里面真正对分类有决定性影响的数据提取出来.
# =============================================================================

 %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; 
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

n_samples, h, w = faces.images.shape

# =============================================================================
#  每一幅图的尺寸为 [62×47] 
# =============================================================================

 fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])

#将数据分为训练和测试数据集
    
 from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)

print(Xtrain.shape)
print(Xtest.shape)

# =============================================================================
#  将整个图像展平为一个长度为3000左右的一维向量, 然后使用这个向量做为特征.
#  PCA(主成分分析), 将一副图像转换为一个长度为更短的(150)向量.
# =============================================================================

 from sklearn.svm import SVC
from sklearn.decomposition import PCA

n_components = 150
print("从%d张人脸图片中提取出top %d eigenfaces" % (n_components, Xtrain.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True, random_state=42).fit(Xtrain)

eigenfaces = pca.components_.reshape((n_components, h, w))

print("将输入数据投影到eigenfaces的标准正交基")

Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

# =============================================================================
#  提取出top 1011 eigenfaces
# 将输入数据投影到eigenfaces的标准正交基
# =============================================================================

#先使用线性svm尝试，作为baseline

 svc = SVC(kernel='linear',C=10)
svc.fit(Xtrain_pca, ytrain)
yfit = svc.predict(Xtest_pca) 

 from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))

#调参:通过交叉验证寻找最佳的kernel，和其他参数，其中： C (控制间隔的大小)

from sklearn.model_selection import GridSearchCV


#param_grid = {'kernel': ('linear', 'rbf','poly'),'C': [1, 5, 10, 50],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = [
    {'kernel': ['linear'], 'C': [1, 5, 10, 50]},
    {'kernel': ['rbf'], 'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]},
    {'kernel': ['poly'], 'C': [1, 5, 10, 50], 'degree':[2,3,4], 'gamma': ['auto']}
]
grid = GridSearchCV(SVC(class_weight='balanced'), param_grid,cv=5)

%time grid.fit(Xtrain_pca, ytrain)
print(grid.best_estimator_)
print(grid.best_params_)

model = grid.best_estimator_
yfit = model.predict(Xtest_pca)

# =============================================================================
# 使用训练好的SVM做预测
# =============================================================================

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);

#生成性能报告
 from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))

#混淆矩阵
 from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');



