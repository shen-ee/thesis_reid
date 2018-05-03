import  numpy as np
from decimal import Decimal

class LSHparam:
    def __init__(self):
        self.nbits = -1
        self.w =[]

def trainLSH(X,LSHparam):
    Ndim = len(X[0])
    nbits = LSHparam.nbits
    LSHparam.w = np.random.randn(Ndim,nbits)
    print("权重为：")
    print(LSHparam.w)

def compressLSH(X,LSHparam):
    U = np.dot(X,LSHparam.w);
    print("矩阵相乘结果为：")
    print(U)
    U = U>0
    print("二进制结果为：")
    print(U.astype(int))

def readFeature(filename):
    features = []
    for line in open("queryfeature.txt"):
        feature = []
        info = line.split(" ")[:2048]
        for item in info:
            feature.append(float(item))
        features.append(feature)
    print(features)
    return features


X = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
param = LSHparam()
param.nbits = 8

print("原向量维度为："+str(len(X[0])))
print("目标向量维度为："+str(param.nbits))

trainLSH(X,param)
compressLSH(X,param)

readFeature("queryfeature.txt")



