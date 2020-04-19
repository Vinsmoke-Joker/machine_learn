import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # 平均轮廓系数

# 1.获取数据
aisles = pd.read_csv('./data/instacart/aisles.csv')
order_product = pd.read_csv('./data/instacart/order_products__prior.csv')
orders = pd.read_csv('./data/instacart/orders.csv')
products = pd.read_csv('./data/instacart/products.csv')
# 2.合并表格
table = pd.merge(order_product,products,on=['product_id','product_id'])
table = pd.merge(table,orders,on=["order_id", "order_id"])
table = pd.merge(table, aisles, on=["aisle_id", "aisle_id"])
# 建立交叉表
table = pd.crosstab(table["user_id"], table["aisle"])
# 3.特征选择-PCA
transform = PCA(n_components=0.9)
data = transform.fit_transform(table)
# 便于演示 截取部分数据
data = data[:1000,:]
# 4.机器学习
estimator = KMeans(n_clusters=8)
estimator.fit(data)
# 模型评估
y_predict = estimator.predict(data)
res = silhouette_score(data,y_predict)
print('平均轮廓系数为：\n',res)