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
    #print("矩阵相乘结果为：")
    #print(U)
    U = U>0
    #print("二进制结果为：")
    #print(U.astype(int))
    return U.astype(int)

def readFeature(filename):
    features = []
    for line in open(filename):
        feature = []
        info = line.split(" ")[:2048]
        for item in info:
            feature.append(float(item))
        features.append(feature)
    #print(len(features))
    return features

def dis(A,B):  # 马氏距离
    C=[]
    for i in range(len(A)):
        C.append(A[i]^B[i])
    return sum(C)


param = LSHparam()
param.nbits = 512

querydata = readFeature("queryfeature.txt")

trainLSH(querydata,param)

query = compressLSH(querydata,param)

testsetdata = readFeature("testfeature.txt")

gallery = compressLSH(testsetdata,param)

print("原向量维度为："+str(len(querydata[0])))
print("目标向量维度为："+str(param.nbits))

minnum = -1
mindis = 99999

for i,item in enumerate(gallery):
    distance = dis(item,query[0])
    if(distance<mindis):
        minnum = i
        mindis = distance
    print(str(i)+":"+str(distance))

print("号码为："+str(minnum))


