# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:08:08 2017
@author: taoyanqi
"""

from __future__ import print_function  
from sklearn import datasets
from sklearn.model_selection import cross_val_score 
from sklearn import preprocessing
import time
import numpy as np
import argparse

def knn_classifier(X,y):
    start_time = time.time()
    print("start knn algorithm ...")
    from sklearn.neighbors import KNeighborsClassifier 
    # 选择超参数k
    K = range(1,10,1)
    for k in K: 
        knn_estimator = KNeighborsClassifier(n_neighbors=k)  
        # 10折交叉验证
        scores = cross_val_score(knn_estimator, X, y, cv =10, scoring = 'accuracy')
        print ("k:%d,accuracy:%.2f%%" %(k,100*scores.mean()))  
    print("knn complete cost:%.2fs time" %(time.time()-start_time))  

def svm_classifier(X,y):
    start_time = time.time()
    print("start svm algorithm ...")
    from sklearn.svm import SVC  
    svm_estimator = SVC(kernel='rbf', probability=True)  
    C=[1e-3, 1e-2, 1e-1, 1, 10,100]
    Gamma = [0.1,0.001, 0.0001]
    for c in C:
        for gamma in Gamma:
            svm_estimator = SVC(C=c,gamma=gamma,kernel='rbf', probability=True)  
            scores = cross_val_score(svm_estimator, X, y, cv = 10, scoring = 'accuracy')
            print ("C:%.3f,gamma:%.4f,accuracy:%.2f%%" %(c,gamma,100*scores.mean())) 
    print("svm complete cost:%.2fs time" %(time.time()-start_time))  
    
def logistic_regression_classifier(X,y):
    start_time = time.time()
    print("start logistic regression algorithm ...")
    from sklearn.linear_model import LogisticRegression  
    lr_estimator = LogisticRegression()
    scores = cross_val_score(lr_estimator, X, y, cv = 10, scoring = 'accuracy')
    print ("accuracy:%.2f%%" % (100*scores.mean()))
    print("logistic regression complete cost:%.2fs time" %(time.time()-start_time))  
    
def random_forest_classifier(X,y):
    start_time = time.time()
    print("start random forest algorithm ...")
    from sklearn.ensemble import RandomForestClassifier 
    num_tree = range(10,200,20)
    for tree in num_tree:
        rf_estimator = RandomForestClassifier(n_estimators=tree)  
        scores = cross_val_score(rf_estimator, X, y, cv =10, scoring = 'accuracy')
        print ("num of tree:%d,accuracy:%.2f%%" %(tree,100*scores.mean()))  
    print("random forest complete cost:%.2fs time" %(time.time()-start_time)) 

def gradient_boosting_classifier(X,y):
    start_time = time.time()
    print("start gradient boosting algorithm ...")
    from sklearn.ensemble import GradientBoostingClassifier  
    num_tree = range(10,200,20)
    for tree in num_tree:
        gbdt_estimator = GradientBoostingClassifier(n_estimators=tree)  
        scores = cross_val_score(gbdt_estimator, X, y, cv =10, scoring = 'accuracy')
        print ("num of tree:%d,accuracy:%.2f%%" %(tree,100*scores.mean()))  
    print("gradient boosting complete cost:%.2fs time" %(time.time()-start_time)) 
   
def read_data(train_data):  
    if train_data:
        dataset = np.loadtxt(train_data,delimiter='\t')
        X = dataset[:,:-1]
        y = dataset[:,-1]

    else:
        #下载示例数据
        iris = datasets.load_iris()
        X = iris.data 
        y = iris.target
    # 进行数据的归一化处理        
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    return X,y

def add_arguments(parse):
    parse.add_argument("--train_data",type=str,default=None,help="input your train data path")

def main():
    X,y= read_data(flags.train_data)
    knn_classifier(X,y)
    svm_classifier(X,y)
    logistic_regression_classifier(X,y)
    random_forest_classifier(X,y)
    gradient_boosting_classifier(X,y)
    
    
if __name__ == "__main__":
    parse  = argparse.ArgumentParser()
    add_arguments(parse)
    flags,unparsed = parse.parse_known_args()
    main()