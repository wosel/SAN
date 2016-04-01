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
    questionWordSet = set()
    answerWordSet = set()
    maxAnswerLen = -1
    maxQuestionLen = -1
    questionWordSet.add('?')
    answerWordSet.add('?')
    for line in open(filename, 'r'):
        if (qLimit > 0 and len(questionList) >= qLimit):
            break
        allWords = line.split()
        #for w in allWords:
         #   wordSet.add(w)
        if(allWords[-1]) != '?': # this is an answer
            for w in range(len(allWords)):
                answerWordSet.add(allWords[w])

            tmpAnswer = allWords

            question = qa(qID=qID, wordList=tmpQuestion, answerList=tmpAnswer, imageID=tmpImgID)
            questionList = [question] + questionList
            if len(allWords) > maxAnswerLen:
                maxAnswerLen = len(allWords)

        else:
            qID += 1
            tmpQuestion = allWords[:-4]
            for w in range(len(allWords)-4):
                questionWordSet.add(allWords[w])

            tmpQuestion = tmpQuestion + ['?']

            imString = allWords[-2]
            imString = re.sub('image', '', imString)
            tmpImgID = int(imString)
            if (len(allWords) - 3) > maxQuestionLen:
                maxQuestionLen = len(allWords) - 3
    questionList.reverse()
    return [questionList, questionWordSet, maxQuestionLen, answerWordSet, maxAnswerLen]

def fillQs(qList, questionWordSet, maxQL, answerWordSet, maxAL):

    questionWordDict = {}

    i = 0;
    for w in questionWordSet:
        questionWordDict[w] = i
        i += 1

    answerWordDict = {}
    answerReverseDict = {}
    i = 0;
    for w in answerWordSet:
        answerWordDict[w] = i
        answerReverseDict[i] = w
        i += 1

    for q in qList:
        if (q.qID % 100 == 0): print(q.qID)
        q.QIdxList = []
        #q.AIdxList = []
        q.oneHotAnswer=np.zeros((len(answerWordDict), maxAL))
        pos = 0
        for w in q.question:
            id = questionWordDict[w]
            q.QIdxList += [id]
            #q.oneHotQuestion[id, pos] = 2.
            #pos += 1
        pos = 0
        for w in q.answer:
            id = answerWordDict[w]
            #q.AIdxList += [id]
            q.oneHotAnswer[id, pos] = 1.
            pos += 1

    return [qList, answerReverseDict]


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

    [_, questionWordSet, maxQL, answerWordSet, maxAL] = loadQuestions(qFolder + "/" + qFullFile)

    [trainQList, _, _, _, _] = loadQuestions(qFolder + "/" + qTrainFile, qLimit)
    [testQList, _, _, _, _] = loadQuestions(qFolder + "/" + qTestFile, qLimit)

    [trainQList, trainRevDict] = fillQs(trainQList, questionWordSet, maxQL, answerWordSet, maxAL)
    trainSet = questionDataset('trainSet')
    trainSet.questionList = trainQList;
    trainSet.count = len(trainQList)
    trainSet.questionDictSize = len(questionWordSet)
    trainSet.qLength = maxQL
    trainSet.answerDictSize = len(answerWordSet)
    trainSet.aLength = maxAL
    trainSet.qMatrix = buildQMatrix(trainQList, maxQL)
    trainSet.aMatrix = buildAMatrix(trainQList, trainSet.answerDictSize, maxAL)

    trainSet.iMatrix = buildIMatrix(trainQList, iFolder)

    [testQList, testRevDict] = fillQs(testQList, questionWordSet, maxQL, answerWordSet, maxAL)
    testSet = questionDataset('testSet')
    testSet.questionList = testQList;
    testSet.count = len(testQList)
    testSet.questionDictSize = len(questionWordSet)
    testSet.qLength = maxQL
    testSet.answerDictSize = len(answerWordSet)
    testSet.aLength = maxAL
    testSet.qMatrix = buildQMatrix(testQList, maxQL)
    testSet.aMatrix = buildAMatrix(testQList, testSet.answerDictSize, maxAL)
    testSet.iMatrix = buildIMatrix(testQList, iFolder)

    return [imageSet, trainSet, testSet, testRevDict]



def augment_data(trainSet, finalCount=10000):
    tmpList = []
    answerCounts = np.sum(trainSet.aMatrix, axis=0)
    tmpCounts = np.zeros(answerCounts.shape)
    maxCount = np.max(answerCounts)

    for q in trainSet.questionList:
        answerIdx = np.argmax(trainSet.aMatrix[q.qID])
        answerCount = answerCounts[answerIdx]
        rep = int(np.floor(maxCount/answerCount))
        for i in range(rep):
            tmpList.append(q.qID)

    finalIdxs = np.random.randint(low=0, high=len(tmpList), size=(finalCount))

    augmentedTrainSet = questionDataset('augmented_trainset')
    augmentedTrainSet.questionList = []
    augmentedTrainSet.count = finalCount
    augmentedTrainSet.questionDictSize = trainSet.questionDictSize
    augmentedTrainSet.qLength = trainSet.qLength
    augmentedTrainSet.answerDictSize = trainSet.answerDictSize
    augmentedTrainSet.aLength = trainSet.aLength

    augmentedTrainSet.qMatrix = np.zeros((finalCount, trainSet.qLength))
    augmentedTrainSet.aMatrix = np.zeros((finalCount, trainSet.answerDictSize))
    augmentedTrainSet.iMatrix = np.zeros((finalCount, 3, 224, 224))

    for fq in range(finalCount):
        q = tmpList[finalIdxs[fq]]
        augmentedTrainSet.questionList.append(trainSet.questionList[q])
        augmentedTrainSet.qMatrix[fq, :] = trainSet.qMatrix[q, :]
        augmentedTrainSet.aMatrix[fq, :] = trainSet.aMatrix[q, :]
        augmentedTrainSet.iMatrix[fq, :, :, :] = trainSet.iMatrix[q, :, :, :]

    return(augmentedTrainSet)
