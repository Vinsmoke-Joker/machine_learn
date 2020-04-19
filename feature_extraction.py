# 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer
# sklearn.base.text.CountVectorizer(stop_words=[])
# 返回词频矩阵
# CountVectorizer.fit_transform(X)
# X:文本或者包含文本字符串的可迭代对象
# 返回值:返回sparse矩阵
# CountVectorizer.get_feature_names() 返回值:单词列表
data = ["life is short,i like python","life is too long,i dislike python"]
transform = CountVectorizer()
res = transform.fit_transform(data)
print("文本特征抽取的结果：\n",res.toarray())
print("返回特征名字：\n", transform.get_feature_names())

# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer
# sklearn.base.DictVectorizer(sparse=True,…)
# DictVectorizer.fit_transform(X)
# X:字典或者包含字典的迭代器返回值
# 返回sparse矩阵
# DictVectorizer.get_feature_names() 返回类别名称
data = [{'city': '北京','temperature':100},
        {'city': '上海','temperature':60},
        {'city': '深圳','temperature':30}]
transform = DictVectorizer(sparse=True)
res = transform.fit_transform(data)
print("字典特征抽取的结果：\n",res)
print("返回特征名字：\n", transform.get_feature_names())

# 中文文本特征抽取
import jieba
datas = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
temp = list()
def cut_words(text):
    # print(jieba.cut(text))
    # print(list(jieba.cut(text)))
    # print(''.join(list(jieba.cut(text))))
    return ' '.join(list(jieba.cut(text)))

for data in datas:
    temp.append(cut_words(data))
transform = CountVectorizer()
res = transform.fit_transform(temp)
print('中文文本特征抽取结果:\n',res.toarray())
print('返回特征名字：\n',transform.get_feature_names())

#  Tf-idf文本特征提取
from sklearn.feature_extraction.text import TfidfVectorizer

transform = TfidfVectorizer(stop_words=['一种'])
res = transform.fit_transform(temp)
print('TF-idf文本特征抽取结果:\n',res.toarray())
print('返回特征名字：\n',transform.get_feature_names())