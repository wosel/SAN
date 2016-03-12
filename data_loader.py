import pandas as pd;
import numpy as np;
import scipy.ndimage as spi
from scipy.misc import imresize
from scipy.sparse import csc_matrix, csr_matrix
import re
import sys
from os import listdir
from os.path import isfile, join


class qa:
    def __init__(self):
        self.qID = -1
        self.question = []
        self.oneHotQuestion
        self.oneHotAnswer
        self.imageID = -1
        self.answer = []
    def __init__(self, qID, wordList, imageID):
        self.qID = qID
        self.question = wordList
        self.imageID = imageID
    def __init__(self, qID, wordList, answerList, imageID):
        self.qID = qID
        self.question = wordList
        self.imageID = imageID
        self.answer = answerList

class imageDataset:
    def __init__(self, name):
        self.name = name

class questionDataset:
    def __init__(self, name):
        self.name = name

def loadQuestions(filename, qLimit=-1):


    qID = -1
    questionList = []

    tmpAnswer = []
    tmpQuestion = []
    tmpImgID = -1
    wordSet = set()
    maxAnswerLen = -1
    maxQuestionLen = -1
    wordSet.add('?')
    for line in open(filename, 'r'):
        if (qLimit > 0 and len(questionList) >= qLimit):
            break
        allWords = line.split()
        #for w in allWords:
         #   wordSet.add(w)
        if(allWords[-1]) != '?': # this is an answer
            for w in range(len(allWords)):
                wordSet.add(allWords[w])

            tmpAnswer = allWords

            question = qa(qID=qID, wordList=tmpQuestion, answerList=tmpAnswer, imageID=tmpImgID)
            questionList = [question] + questionList
            if len(allWords) > maxAnswerLen:
                maxAnswerLen = len(allWords)

        else:
            qID += 1
            tmpQuestion = allWords[:-4]
            for w in range(len(allWords)-4):
                wordSet.add(allWords[w])

            tmpQuestion = tmpQuestion + ['?']

            imString = allWords[-2]
            imString = re.sub('image', '', imString)
            tmpImgID = int(imString)
            if (len(allWords) - 3) > maxQuestionLen:
                maxQuestionLen = len(allWords) - 3
    questionList.reverse()
    return [questionList, wordSet, maxQuestionLen, maxAnswerLen]

def fillQs(qList, fullWordSet, maxQL, maxAL):

    fullWordDict = {}
    fullReverseDict = {}
    i = 0;
    for w in fullWordSet:
        fullWordDict[w] = i
        fullReverseDict[i] = w
        i+=1;
    


    for q in qList:
        if (q.qID % 100 == 0): print(q.qID)
        q.QIdxList = []
        #q.AIdxList = []
        q.oneHotAnswer=np.zeros((len(fullWordDict), maxAL))
        pos = 0
        for w in q.question:
            id = fullWordDict[w]
            q.QIdxList += [id]
            #q.oneHotQuestion[id, pos] = 2.
            #pos += 1
        pos = 0
        for w in q.answer:
            id = fullWordDict[w]
            #q.AIdxList += [id]
            q.oneHotAnswer[id, pos] = 1.
            pos += 1

    return [qList, fullReverseDict]
'''
def fillOneHot(qList, fullWordSet, maxQL, maxAL):

    fullWordDict = {}
    i = 0;
    for w in fullWordSet:
        fullWordDict[w] = i
        i+=1;


    for q in qList:
        if (q.qID % 100 == 0): print(q.qID)
        q.oneHotQuestion = csc_matrix((len(fullWordDict), maxQL), dtype=float)
        q.oneHotAnswer = csc_matrix((len(fullWordDict), maxAL), dtype=float)

        pos = 0
        for w in q.question:
            id = fullWordDict[w]
            q.oneHotQuestion[id, pos] = 1.
            pos += 1
        pos = 0
        for w in q.answer:
            id = fullWordDict[w]
            q.oneHotAnswer[id, pos] = 1.
            pos += 1
    return qList

'''




def buildQMatrix(qList, maxQL):
  
    qMatrix = np.zeros((len(qList), maxQL))
    print(qMatrix.shape)
    for q in qList:
#    qMatrix[q.qID, :] = np.zeros((maxQL,))
        qMatrix[q.qID, 0:len(q.QIdxList)] = q.QIdxList  
    return qMatrix

def buildAMatrix(qList, dictLen, maxAL):
    aMatrix = np.zeros((len(qList), dictLen))
    for q in qList:
        aMatrix[q.qID, :] = q.oneHotAnswer[:, 0]
    return aMatrix


def buildIMatrix(qList, iFolder):
    iMatrix = np.zeros((len(qList), 3, 224, 224))
    for q in qList:
        if (q.qID % 50 == 0):
            print("loaded images for " + str(q.qID) + " questions")
            #iMatrix[q.qID, :, :, :] = images[q.imageID-1, :, :, :]
            iMatrix[q.qID, :, :, :] = loadImage(iFolder, q.imageID)
    return iMatrix
  
def loadImage(folder, imageID):
    imName = "image" + str(imageID) + ".png"
    image =  np.rollaxis(imresize(spi.imread(folder + '/' + imName), size=(224, 224, 3), interp='bilinear'), 2, 0)
    return image

def loadImages(folder, imLimit=-1):
    print("loading from " + folder)
    images = [f for f in listdir(folder) if isfile(join(folder, f))]
    imCt = len(images)
    if (imLimit > 0):
        imCt = imLimit

    imageDataset = np.empty([imCt, 3, 224, 224])
    ct = 0
    # this is sort of stupid and takes about 6GB of RAM, but here we load the whole image dataset into memory
    for imName in images:
        imNameMod = re.sub('image', '', imName)
        imNameMod= re.sub(r'\.png', '', imNameMod)
        i = int(imNameMod)-1
        if (i >= imCt):
            continue
        imageDataset[i, :, :, :] = np.rollaxis(imresize(spi.imread(folder + '/' + imName), size=(224, 224, 3), interp='bilinear'), 2, 0)
        ct += 1
        if (ct % 50 == 0):
            print("loaded " + str(ct) + " images so far ")
        if (ct > imCt):
            break
    return imageDataset;



def load_both(qFolder, qFullFile, qTrainFile, qTestFile, iFolder, imLimit=-1, qLimit=-1 ):

    #imageDS = loadImages(iFolder, imLimit)

    imageSet = imageDataset('allImages')
    #imageSet.dataset = imageDS
    #imageSet.count = imageDS.shape[0]
    #imageSet.imageShape = (imageDS.shape[1], imageDS.shape[2], imageDS.shape[3])

    [_, fullWordSet, maxQL, maxAL] = loadQuestions(qFolder + "/" + qFullFile)

    [trainQList, _, _, _] = loadQuestions(qFolder + "/" + qTrainFile, qLimit)
    [testQList, _, _, _] = loadQuestions(qFolder + "/" + qTestFile, qLimit)

    [trainQList, trainRevDict] = fillQs(trainQList, fullWordSet, maxQL, maxAL)
    trainSet = questionDataset('trainSet')
    trainSet.questionList = trainQList;
    trainSet.count = len(trainQList)
    trainSet.dictSize = len(fullWordSet)
    trainSet.qLength = maxQL
    trainSet.aLength = maxAL
    trainSet.qMatrix = buildQMatrix(trainQList, maxQL)
    trainSet.aMatrix = buildAMatrix(trainQList, trainSet.dictSize, maxAL)
    
    trainSet.iMatrix = buildIMatrix(trainQList, iFolder)
    
    [testQList, testRevDict] = fillQs(testQList, fullWordSet, maxQL, maxAL)
    testSet = questionDataset('testSet')
    testSet.questionList = testQList;
    testSet.count = len(testQList)
    testSet.dictSize = len(fullWordSet)
    testSet.qLength = maxQL
    testSet.aLength = maxAL
    testSet.qMatrix = buildQMatrix(testQList, maxQL)
    testSet.iMatrix = buildIMatrix(testQList, iFolder)
    testSet.aMatrix = buildAMatrix(testQList, testSet.dictSize, maxAL)

    return [imageSet, trainSet, testSet, testRevDict]

