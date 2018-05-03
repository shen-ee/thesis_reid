import  numpy as np
import time

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

def dist(A,B):  # 马氏距离
    A = np.array(A)
    B = np.array(B)
    C = A^B
    return sum(C)

def rank1(PersonID,PhotoID,param):
    minnum = -1
    mindis = 99999
    querydata = readFeature("features/" + str(PersonID) + "_"+str(PhotoID)+".txt")
    gallerydata = readFeature("testfeature.txt")
    query = compressLSH(querydata, param)
    gallery = compressLSH(gallerydata, param)
    for i, item in enumerate(gallery):
        distance = dist(item, query[0])
        if (distance < mindis):
            minnum = i
            mindis = distance
    if(minnum==PersonID):
        return 1
    else:
        return 0


param = LSHparam()
param.nbits = 512
querydata = readFeature("features/0_0.txt")

trainLSH(querydata,param)

total = 0
start = time.time()
for PersonID in range(100):
    total += rank1(PersonID,0,param)

end = time.time()
print(total/100)
print( "所需时间：" + str(end-start) + "s\n")
