from numpy import * 
import operator
from os import listdir


#create function
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classfy(X,DataSet,Labels,k):
    size=DataSet.shape[0]
    diffMat=tile(X,(size,1))-DataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5 
    sortedIndex=distances.argsort()
    Count={}
    for i in range(k):
        vote=Labels[sortedIndex[i]]
        Count[vote]=Count.get(vote,0)+1
    sortCount=sorted(Count.items(),key=lambda x:x[1],reverse=True)
    return sortCount[0][0]   


#
def file2matrix(filename):
    fr=open(filename)
    numArray=fr.readlines()
    numLine=len(numArray)
    Mat=zeros((numLine,3))
    labels=[]
    index=0
    for line in numArray:
        line=line.strip()
        lineFromLine=line.split('\t')
        Mat[index,:]=lineFromLine[0:3]
        labels.append(int(lineFromLine[-1]))
        index+=1
    return Mat,labels

# normize
def norm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    m=dataSet.shape[0]
    diff=dataSet-tile(minVals,(m,1))
    normMat=diff/tile(ranges,(m,1))
    return normMat,minVals,ranges
    
# test
def classfyTest():
    rate=0.1
    mat,labels=file2matrix('datingTestSet2.txt')
    normMat,minVals,ranges=norm(mat)
    m=normMat.shape[0]
    errCount=0.0
    numTest=int(m*rate)
    for i in range(numTest):
        res=classfy(normMat[i,:],normMat[numTest:m,:],labels[numTest:m],3)
        print "the clssifier came back is : %d,the real is %d" % (res,labels[i])
        if (res!=labels[i]):
            errCount+=1.0
    print "the total error is %d ,the rate is %f" %(int(errCount),errCount/float(numTest))

#if __name__ == '__main__':
#    classfyTest()


#
def img2vector(filename):
    retVec=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            retVec[0,i*32+j]=lineStr[j]
    return retVec
#
def handWritingClassTest():
    labels=[]
    fileList=listdir('trainingDigits')            #it is a dir,and return a list of filename
    m=len(fileList)
    mat=zeros((m,1024))
    for i in range(m):
        fileName=fileList[i]
        fileStr=fileName.split('.')[0]            # split the filename
        className=int(fileStr.split('_')[0])
        labels.append(className)
        mat[i,:]=img2vector('trainingDigits/%s' % fileName)
    errCount=0.0
    testFileList=listdir('testDigits')
    mTest=len(testFileList)
    for i in range(mTest):
        fileName=testFileList[i]
        fileStr=fileName.split('.')[0]
        className=int(fileStr.split('_')[0])
        test=img2vector('testDigits/%s' % fileName)
        res=classfy(test,mat,labels,3)
        print "the clssfier came back %d,the real is %d" % (res,className)
        if (res!=className):
            errCount+=1.0
    print "the total error is %d,the rate is %f" %(int(errCount),errCount/float(mTest))

#################################################################################################
if __name__ == '__main__':
	#classfyTest()                     #you can also run this program
    handWritingClassTest()

    







