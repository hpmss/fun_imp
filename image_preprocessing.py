import os
import tensorflow as tf
import numpy as np
import cv2
import threading

PATH = "D:\\hpms\\mlearn\\vision\\models\\ConvolutionalNetwork\\data_sets\\train_data"
SAVE_PATH = "D:\\hpms\\mlearn\\vision\\models\\ConvolutionalNetwork\\data_sets\\saved_data"

WIDTH = 224
HEIGHT = 224

SIZE = 6000
MINI_SIZE = 200

CAT_LABEL = 0
DOG_LABEL = 1

CAT_LABEL_STRING = 'cat'
DOG_LABEL_STRING = 'dog'

"""
    Notes:
        -> This implementation is used for loading the dataset "DOGS vs CATS" from Kaggle (25000 samples)
        -> This currently only works if SIZE and MINI_SIZE are a multiple of 2
        (Because i didnt test other SIZE and MINI_SIZE yet (´･ω･`) but i doubt they works... )

        -> My implementation is kind of limited since the dataset is already sorted so i only load
        -> cats first then i load for dogs but the GENERAL IDEAS is shown below.

    The algorithm uses MINI_SIZE to check for whether the current TRAIN_DATA's size is equal to MINI_SIZE
    There will be two cases:
        -> '*' denotes characters...
        -> If there is no '*_temp.npy' , it will save for the first time
        -> Otherwise it will load '*_temp.npy' and merge it with the current TRAIN_DATA
        -> Then save it by overwriting the current '*_temp.npy'
        -> Resetting TRAIN_DATA = None ( makes it size = 0 )
    This works out since:
        -> For personal reason (depend on each computer speculation) , my computer loads fast for 200 images.
        -> As the TRAIN_DATA size grows , it is impossible to load more images at fast pace.
        -> So one way is to reset TRAIN_DATA back to 0.
        -> This makes it load fast again for 200 images.
        -> Merging previous datas and current datas as the size grows could be a little bit long.
        -> But this is efficient for low-end CPU and GPU.
"""

def load_data():

    TRAIN_DATA = None
    TRAIN_LABEL = None
    try:
        TRAIN_DATA = np.load(SAVE_PATH + "\\train_data.npy")
        TRAIN_LABEL = np.load(SAVE_PATH + "\\train_label.npy")
        if TRAIN_DATA.shape[0] == SIZE and TRAIN_LABEL.shape[0] == SIZE:
            print('Image datas already loaded.')
            return (TRAIN_DATA,TRAIN_LABEL)
        else:
            TRAIN_LABEL = np.array([])
            TRAIN_DATA = None
    except FileNotFoundError as err:
        TRAIN_LABEL = np.array([])

    dog = 0
    cat = 0

    print('Loading image datas...')
    for i,filename in enumerate(os.listdir(PATH)):
        FILE = PATH + "\\" + filename
        NAME = filename.split(".")
        img_data = np.asarray(cv2.resize(cv2.cvtColor(cv2.imread(FILE),cv2.COLOR_BGR2RGB),dsize= (WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC))
        img_data = np.expand_dims(img_data,axis=0)

        if cat != SIZE / 2:
            if NAME[0] == CAT_LABEL_STRING:
                cat += 1
                print('Cat file number: ',cat)
                if TRAIN_DATA is None:
                    TRAIN_DATA = img_data
                else:
                    TRAIN_DATA = np.vstack([TRAIN_DATA,img_data])
                TRAIN_LABEL = np.append(TRAIN_LABEL,[CAT_LABEL])
            else:
                print('Skipped file number: ',i)
                continue
            if TRAIN_DATA is not None and TRAIN_DATA.shape[0] == MINI_SIZE:
                if os.path.isfile(SAVE_PATH + "\\train_data_cat_temp.npy"):
                    print("Merge mini-sized train datas to current datas...")
                    temp = np.load(SAVE_PATH + "\\train_data_cat_temp.npy")
                    merged = np.vstack([temp,TRAIN_DATA])
                    np.save(SAVE_PATH + "\\train_data_cat_temp",merged)
                    TRAIN_DATA = None
                else:
                    print('Saving temporary train datas to train_data_cat_temp.npy... -> shape%s' % (str(TRAIN_DATA.shape)))
                    np.save(SAVE_PATH + "\\train_data_cat_temp",TRAIN_DATA)
                    TRAIN_DATA = None

        elif dog != SIZE / 2:
            if NAME[0] == DOG_LABEL_STRING:
                dog += 1
                print('Dog file number: ',dog)
                if TRAIN_DATA is None:
                    TRAIN_DATA = img_data
                else:
                    TRAIN_DATA = np.vstack([TRAIN_DATA,img_data])
                TRAIN_LABEL = np.append(TRAIN_LABEL,[DOG_LABEL])
            else:
                print('Skipped file number: ',i)
                continue
            if TRAIN_DATA is not None and TRAIN_DATA.shape[0] == MINI_SIZE:
                if os.path.isfile(SAVE_PATH + "\\train_data_dog_temp.npy"):
                    print("Merging mini-sized train datas to current datas...")
                    temp = np.load(SAVE_PATH + "\\train_data_dog_temp.npy")
                    merged = np.vstack([temp,TRAIN_DATA])
                    np.save(SAVE_PATH + "\\train_data_dog_temp",merged)
                    TRAIN_DATA = None
                else:
                    print('Saving temporary train datas to train_data_dog_temp.npy... -> shape%s' % (str(TRAIN_DATA.shape)))
                    np.save(SAVE_PATH + "\\train_data_dog_temp",TRAIN_DATA)
                    TRAIN_DATA = None
        else:
            break

    print('Saving train labels to train_label.npy... -> shape%s' % (str(TRAIN_LABEL.shape)))
    np.save(SAVE_PATH + "\\train_label",TRAIN_LABEL)
    print('Merging temporary train datas and saving to train_data.npy... -> shape%s' % (str(TRAIN_DATA.shape)))
    cat_data = np.load(SAVE_PATH + "\\train_data_cat_temp.npy")
    dog_data = np.load(SAVE_PATH + "\\train_data_dog_temp.npy")
    TRAIN_DATA = np.vstack([cat_data,dog_data])
    np.save(SAVE_PATH + "\\train_data",TRAIN_DATA)
    print('Clean up temporary datas...')
    os.remove(SAVE_PATH + "\\train_data_cat_temp.npy")
    os.remove(SAVE_PATH + "\\train_data_dog_temp.npy")
    return (TRAIN_DATA,TRAIN_LABEL)

load_data()
