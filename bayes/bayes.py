from numpy import *

#create a vocablist of set ,word can only exit once

def createVocabList(dataSet):
    vocabSet=set([])
    for docment in dataSet:
        vocabSet=vocabSet| set(docment) #union of tow sets
    return list(vocabSet) #convet if to list 


def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else: print ("the word is not in my vocabulry")
    return returnVec

# tranin algorithm
# the p1Num is mean claclualte in 1 class evrey word contain weight
def train(trainMat,trainGategory):
    numTrain=len(trainMat)
    numWords=len(trainMat[0])  #is vocabulry length
    pAbusive=sum(trainGategory)/float(numTrain)
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrain):
        if trainGategory[i] == 1:
            p1Num += trainMat[i] 
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom +=sum(trainMat[i])
    p1Vec=log(p1Num/p1Denom)
    p0Vec=log(p0Num/p0Denom)
    return p0Vec,p1Vec,pAbusive
# classfy funtion
def classfy(vec2classfy,p0Vec,p1Vec,pClass1):
    p1=sum(vec2classfy*p1Vec)+log(pClass1)
    p0=sum(vec2classfy*p0Vec)+log(1-pClass1)
    if p1 > p0:
        return 1;
    else:
        return 0

# split the big string
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#spam email classfy
def spamTest():
    fullTest=[];docList=[];classList=[]
    for i in range(1,26): #it only 25 doc in every class
        wordList=textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)   # create vocabulry
    trainSet=range(50);testSet=[]
#choose 10 sample to test ,it index of trainMat
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSet)))#num in 0-49
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat=[];trainClass=[]
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0,p1,pSpam=train(array(trainMat),array(trainClass))
    errCount=0
    for docIndex in testSet:
        wordVec=bagOfWords2Vec(vocabList,docList[docIndex])
        if classfy(array(wordVec),p0,p1,pSpam) != classList[docIndex]:
            errCount +=1
            print ("classfication error"), docList[docIndex]
           
    print ("the error rate is ") , float(errCount)/len(testSet)

if __name__ == '__main__':
    spamTest()
         





