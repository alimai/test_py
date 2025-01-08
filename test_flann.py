# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:03:33 2024

@author: maishengli

1. 
'linear'：线性搜索，适用于数据集非常小的情况。
2. 
'kdtree'：KD树（K-Dimensional Tree），适用于低维数据。
3. 
'kmeans'：K均值树，适用于高维数据。
4. 
'composite'：复合索引，结合了KD树和K均值树的优点。
5. 
'lsh'：局部敏感哈希（Locality-Sensitive Hashing），适用于高维稀疏数据。
6. 
'saved'：加载之前保存的索引。
"""


import numpy as np
import pyflann

# 生成一些随机数据
np.random.seed(0)
data = np.random.rand(1000, 128)  # 1000个128维的数据点
query = np.random.rand(5, 128)   # 5个128维的查询点

flann = pyflann.FLANN()
params = flann.build_index(data, algorithm = "kmeans", trees=4 )
result, dists = flann.nn_index(query, num_neighbors=1)

print("最近邻索引：", result)
print("最近邻距离：", dists)