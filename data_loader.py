import pandas as pd;
import numpy as np;
import scipy.ndimage as spi
import re

from os import listdir
from os.path import isfile, join

daquarFolder = 'C:/daquar'
daquarImageFolder = daquarFolder + '/nyu_depth_images'
files = [f for f in listdir(daquarFolder) if isfile(join(daquarFolder, f))]
images = [f for f in listdir(daquarImageFolder) if isfile(join(daquarImageFolder, f))]


image_dataset = np.empty([len(images), 425, 560, 3])
ct = 0

# this is sort of stupid and takes about 6GB of RAM, but here we load the whole image dataset into memory
for imName in images:
    imNameMod = re.sub('image', '', imName)
    imNameMod= re.sub(r'\.png', '', imNameMod)
    i = int(imNameMod)-1
    image_dataset[i,:,:,:] = spi.imread(daquarImageFolder+'/'+imName)
    ct += 1
    if (ct%20 == 0):
        print(ct)





