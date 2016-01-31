import pandas as pd;
import numpy as np;
import scipy.ndimage as spi
import re
import sys
from os import listdir
from os.path import isfile, join

class qa:
    def __init__(self):
        self.qID = -1
        self.question = []
        self.oneHotWords = []
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

    for line in open(filename, 'r'):
        allWords = line.split()
        for w in allWords:
            wordSet.add(w)
        if(allWords[-1]) != '?': # this is an answer
            tmpAnswer = allWords
            question = qa(qID=qID, wordList=tmpQuestion, answerList=tmpAnswer, imageID=tmpImgID)
            questionList = [question] + questionList
        else:
            qID += 1
            tmpQuestion = allWords[:-4]
            imString = allWords[-2]
            imString = re.sub('image', '', imString)
            tmpImgID = int(imString)
    return [questionList.reverse(), wordSet]

[questionList, fullWordSet] = loadFile(daquarFolder+'/qa.894.raw.txt')

print(len(fullWordSet))
fullWordDict = {}
i = 0;
for w in fullWordSet:
    fullWordDict[w] = i
    i+=1;
print(fullWordDict['a'])





