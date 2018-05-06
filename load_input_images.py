import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


def load_input(path='../Mikro_markiert'):
        
    img_names = (i for i in os.listdir(path))    
    X,y = np.empty((4,600,800,3), dtype=np.uint8), np.empty((4,600,800,1), dtype=np.uint8)
    c = 0
    for name in img_names:
        img = cv2.imread(os.path.join('../Mikro_markiert', name))    
        ret, mask = cv2.threshold(img[:,:,2], 250, 255, cv2.THRESH_BINARY)

        original = cv2.imread(os.path.join('../Mikro_unmarkiert', name.replace('_markiert', '')))
        
        X[c] = cv2.resize(original, (800,600))
        y[c,:,:,0] = cv2.resize(mask, (800,600))
        
        c += 1
        if c == 4:
            yield X,y
            c = 0

        
        
        
if __name__=='__main__':
        
        
    gen = load_input()
    
    for n,(x,y) in enumerate(gen):
        cv2.imshow('orig', y[n])