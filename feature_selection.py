# 降维 -低方差过滤，PCA
# 1.低方差过滤
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
# data = pd.read_csv('./data/factor_returns.csv')
data = [[2,8,4,5],
[6,3,0,5],
[5,4,9,5]]
transform = VarianceThreshold(threshold=0.0)
# data = transform.fit_transform(data.iloc[:,1:10])
data = transform.fit_transform(data)
print("低方差过滤的结果：\n", data)
print("形状：\n", data.shape)
# 2.主成分分析-PCA
from sklearn.decomposition import PCA
data = [[2,8,4,5],
[6,3,0,8],
[5,4,9,1]]
# n_components 保留百分之多少信息 一般90%-95%即0.9-0.95
transform = PCA(n_components=0.9)
data = transform.fit_transform(data)
print("PCA的结果：\n", data)
print("形状：\n", data.shape)