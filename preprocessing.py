from sklearn.preprocessing import MinMaxScaler
data = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]
# 归一化处理 缺点：对于异常点过于敏感
transform = MinMaxScaler()
res = transform.fit_transform(data)
print('归一化结果为：\n',res)
print('指定范围2-3内进行归一化\n')
transform_2 = MinMaxScaler(feature_range=(2,3))
res = transform_2.fit_transform(data)
print('归一化结果为：\n',res)

# 标准化
from sklearn.preprocessing import StandardScaler
transform = StandardScaler()
res = transform.fit_transform(data)
print('标准化结果为：\n',res)

# 缺失值处理
import numpy as np
from sklearn.preprocessing import Imputer
data = [[90,2,np.nan,40],[60,np.nan,15,45],[75,3,13,np.nan]]
# 对于list series类型，通过Imputer处理
transform = Imputer(missing_values=np.nan,strategy='mean',axis=0)
data = transform.fit_transform(data)
print('处理后数据为\n',data)
# 对于DataFrame 缺失值，需要通过replace处理
# data = data.replace(toreplace='?',value=np.nan)
# data = data.dropna
# 单独列的Nan处理
# x['age'].fillna(x['age'].mean(), inplace=True)
