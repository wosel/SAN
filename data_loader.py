import pandas as pd;
import numpy as np;
import scipy.ndimage as spi
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



daquarFolder = 'C:/daquar'
daquarImageFolder = daquarFolder + '/nyu_depth_images'
files = [f for f in listdir(daquarFolder) if isfile(join(daquarFolder, f))]
images = [f for f in listdir(daquarImageFolder) if isfile(join(daquarImageFolder, f))]


image_dataset = np.empty([len(images), 425, 560, 3])
ct = 0

# this is sort of stupid and takes about 6GB of RAM, but here we load the whole image dataset into memory
'''
for imName in images:
    imNameMod = re.sub('image', '', imName)
    imNameMod= re.sub(r'\.png', '', imNameMod)
    i = int(imNameMod)-1
    image_dataset[i,:,:,:] = spi.imread(daquarImageFolder+'/'+imName)
    ct += 1
    if (ct%20 == 0):
        print(ct)
'''





def loadFile(filename):


    qID = -1
    questionList = []

    tmpAnswer = []
    tmpQuestion = []
    tmpImgID = -1
    wordSet = set()
    maxAnswerLen = -1
    maxQuestionLen = -1
    for line in open(filename, 'r'):
        allWords = line.split()
        for w in allWords:
            wordSet.add(w)
        if(allWords[-1]) != '?': # this is an answer
            tmpAnswer = allWords

            question = qa(qID=qID, wordList=tmpQuestion, answerList=tmpAnswer, imageID=tmpImgID)
            questionList = [question] + questionList
            if len(allWords) > maxAnswerLen:
                maxAnswerLen = len(allWords)

        else:
            qID += 1
            tmpQuestion = allWords[:-4]
            tmpQuestion = tmpQuestion + ['?']

            imString = allWords[-2]
            imString = re.sub('image', '', imString)
            tmpImgID = int(imString)
            if (len(allWords) - 3) > maxQuestionLen:
                maxQuestionLen = len(allWords) - 3
    questionList.reverse()
    return [questionList, wordSet, maxQuestionLen, maxAnswerLen]

[qList, fullWordSet, maxQL, maxAL] = loadFile(daquarFolder + '/qa.37.raw.txt')

print(len(fullWordSet))
fullWordDict = {}
i = 0;
for w in fullWordSet:
    fullWordDict[w] = i
    i+=1;

print("maxAL: " + str(maxAL))
print("maxQL: " + str(maxQL))

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

print (qList[0].oneHotQuestion)
print (qList[0].oneHotQuestion.toarray())
print (qList[0].question)
print (qList[0].oneHotAnswer)
print (qList[0].oneHotAnswer.toarray())
print (qList[0].answer)



