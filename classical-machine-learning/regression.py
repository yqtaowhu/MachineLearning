"""
@author: taoyanqi
"""
from __future__ import print_function  
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_data(train_data):  
    dataset = np.loadtxt(train_data,delimiter='\t')
    X = dataset[:,:-1]
    y = dataset[:,-1]
    """
    #you can preprocessing the features neccessary
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    """
    return X,y

def print_function(string,scores,train_mses,test_mses):
    print("%s start:" % string)
    print("r2 is:%f" % np.array(scores).mean())
    print("train mse is%f" % np.array(train_mses).mean())
    print("test mse is%f" % np.array(test_mses).mean())
    print()

def k_fold_cross_valadation(num,k,estimator,X,y):
    """
    num is the train data total;
    k is k-fold
    estimator is which algorithm to use
    X is a matrix,which represention fetures
    y is label 
    """
    kf = KFold(num,k)
    scores= []
    test_mses = []
    train_mses = []
    for train,test in kf:
        estimator.fit(X[train],y[train])
        score = estimator.score(X[train],y[train])
        y_train_pred = estimator.predict(X[train])
        train_mse = metrics.mean_squared_error(y[train],y_train_pred)
        train_mses.append(train_mse)
        y_pred = estimator.predict(X[test])
        test_mse = metrics.mean_squared_error(y[test],y_pred)
        scores.append(score)
        test_mses.append(test_mse)
    return scores,train_mses,test_mses

def svm_regression(X,y,num,k):
    from sklearn import svm
    """
    # this is to use choose best hyp-parameters
    C=[1e-1, 1, 10,100,1000]
    Gamma = [10,1,0.1,0.001]
    for c in C:
        for gamma in Gamma:
            svm_est = svm.SVR(C=c,gamma=gamma)
            scores,train_mses,test_mses = k_fold_cross_valadation(num,k,svm_est,X,y)
            print("C:%f,gamma:%f" % (c,gamma))
            print_function("svm regression",scores,train_mses,test_mses)
    """       
    svm_estimator = svm.SVR(C=100,gamma=0.1)
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,svm_estimator,X,y)
    print_function("svm regression",scores,train_mses,test_mses)
    """
    # you can use cross_val_predict to do cross valadation
    svm_estimator = svm.SVR(C=20,gamma=10)
    svm_estimator.fit(X,y)
    pred = cross_val_predict(svm_estimator, X, y, cv=10)
    print(metrics.r2_score(y,pred))
    #print(metrics.mean_squared_error(y,svm_estimator.predict(X)))
    print(metrics.mean_squared_error(y,pred))
    """

def linear_regression(X,y,num,k):
    from sklearn import linear_model
    linear_estimator = linear_model.LinearRegression()
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,linear_estimator,X,y)
    print_function("linear regression",scores,train_mses,test_mses)
    # this is using all data to training ,only test
    linear_estimator.fit(X,y)
    print("coef,and intercept",end=' ')
    print(linear_estimator.coef_,end=' '),print(linear_estimator.intercept_),print()


def polynomial_regression(X,y,num,k):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model        
    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(X)
    poly_est = linear_model.LinearRegression()
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,poly_est,X_train,y)
    print_function("poly regression",scores,train_mses,test_mses)
    # this is using all data to training ,only test
    poly_est.fit(X_train,y)
    print("coef,and intercept",end=' ')
    print(poly_est.coef_,end=' '),print(poly_est.intercept_),print()
    
def knn_regression(X,y,num,k):
    from sklearn.neighbors import KNeighborsRegressor
    knn_est = KNeighborsRegressor(n_neighbors=5)
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,knn_est,X,y)
    print_function("knn regression",scores,train_mses,test_mses)

def decision_tree_regression(X,y,num,k):
    from sklearn import tree
    tree_est = tree.DecisionTreeRegressor(max_depth=2)
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,tree_est,X,y)
    print_function("decision tree regression",scores,train_mses,test_mses)

def random_forest_regression(X,y,num,k):
    from sklearn.ensemble import RandomForestRegressor
    rf_est = RandomForestRegressor(max_depth=4,n_estimators=40)
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,rf_est,X,y)
    print_function("random forest regression",scores,train_mses,test_mses)

def gradient_boost_regression(X,y,num,k):
    from sklearn.ensemble import GradientBoostingRegressor
    gbdt_est = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls') 
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,gbdt_est,X,y)
    print_function("gradient boost decision tree regression",scores,train_mses,test_mses)
    
def nn_regression(X,y,num,k):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    scaler.fit(X)
    X_train = scaler.transform(X) 
    nn_est = MLPRegressor(hidden_layer_sizes=(10,4,4),max_iter=100000,
                          learning_rate="adaptive",learning_rate_init=0.001,solver="sgd")
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,nn_est,X,y)
    print_function("neural network regression",scores,train_mses,test_mses)
    
    nn_est = MLPRegressor(hidden_layer_sizes=(4,),max_iter=100000,
                          learning_rate="adaptive",learning_rate_init=0.001,solver="sgd",
                          alpha=0)
    scores,train_mses,test_mses = k_fold_cross_valadation(num,k,nn_est,X_train,y)
    print_function("neural network regression with scale the feature",
                   scores,train_mses,test_mses)

    
def vegetation_index_regression(X,y,num,k):
    """
    this is single feture,only using linear regression and poly regreesion
    """
    
    green,red,red_edge,nir = X[:,0],X[:,1],X[:,2],X[:,3]
    # let 1-d to 2-d
    NDVI = np.array((nir-red)/(nir+red)).reshape(num,1)
    EVI2 = np.array(2.5*(nir-red)/(nir+2.4*red+1)).reshape(num,1)
    CIgreen = np.array(nir/green - 1).reshape(num,1)
    CIred_edge = np.array(nir/red - 1).reshape(num,1)
    NDRE = np.array((nir-red_edge)/(nir+red_edge)).reshape(num,1)
    VARI =np.array((green-red)/(green+red)).reshape(num,1)
    MTCI =np.array((nir-red_edge)/(red_edge-red)).reshape(num,1)
    RDVI =np.array((nir-red)/(nir+red)*(nir-red)).reshape(num,1)

    plt.figure(1)
    plt.subplot(2,2,1),plt.scatter(NDVI,y),plt.legend("NDVI")
    plt.subplot(2,2,2),plt.scatter(EVI2,y),plt.legend("EVI2")
    plt.subplot(2,2,3),plt.scatter(CIgreen,y),plt.legend("CIgreen")
    plt.subplot(2,2,4),plt.scatter(CIred_edge,y),plt.legend("CIred_edge")
    plt.figure(2)
    plt.subplot(2,2,1),plt.scatter(NDRE,y),plt.legend("NDRE")
    plt.subplot(2,2,2),plt.scatter(VARI,y),plt.legend("VARI")
    plt.subplot(2,2,3),plt.scatter(MTCI,y),plt.legend("MTCI")
    plt.subplot(2,2,4),plt.scatter(RDVI,y),plt.legend("RDVI")
    
    """
    # this is a test of all data to training !
    lin_est = linear_model.LinearRegression()
    lin_est.fit(NDVI,y)
    score = lin_est.score(NDVI,y)
    """
    
    vegetation_index = [NDVI,EVI2,CIgreen,CIred_edge,NDRE,VARI,MTCI,RDVI]
    vi_name = ['NDVI','EVI2','CIgreen','CIred_edge','NDRE','VARI','MTCI','RDVI']
    for vi,name in zip(vegetation_index,vi_name):
        print("*************start regression: %s*****************" % name)
        linear_regression(vi,y,num,k)
        polynomial_regression(vi,y,num,k)
    
        
def add_arguments(parse):
    parse.add_argument("--train_data",type=str,default=None,
                       help="input your train data path")
    parse.add_argument("--k_fold",type=int,default=10,
                       help="how many k-fold")
    
def main():  
    test_regression = ['Linear','Poly','KNN','DT','SVM','RF','GBDT','NN','VI']
    regressor = {'SVM':svm_regression,
                 'Linear':linear_regression,
                 'Poly':polynomial_regression,
                 'KNN':knn_regression,
                 'DT':decision_tree_regression,
                 'RF':random_forest_regression,
                 'GBDT':gradient_boost_regression,
                 'NN':nn_regression,
                 'VI':vegetation_index_regression
            }
    print('reading training and testing data...') 
    # X,y= read_data(flags.train_data)
    # you can replace your own data
    X,y= read_data("chl.txt")  
    for reg in test_regression:
        regressor[reg](X,y,np.array(y).shape[0],flags.k_fold)
    
if __name__ == "__main__":
    parse  = argparse.ArgumentParser()
    add_arguments(parse)
    flags,unparsed = parse.parse_known_args()
    main()