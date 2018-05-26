sCHimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


def load_input(path='../Mikro_markiert', bs=4):
        
    img_names = (i for i in os.listdir(path))  
    
    X,y = np.empty((bs,120,160,3), dtype=np.uint8), np.empty((bs,120,160,1), dtype=np.uint8)
    c = 0
    for name in img_names:
        print(name)
        img = cv2.imread(os.path.join('../Mikro_markiert', name))    
        img = cv2.resize(img, (160,120))
        
        
        ret, mask = cv2.threshold(img[:,:,2], 250, 255, cv2.THRESH_BINARY)

        original = cv2.imread(os.path.join('../Mikro_unmarkiert', name.replace('_markiert', '')))
        
        X[c] = cv2.resize(original, (160,120))
        y[c,:,:,0] = mask
        
        c += 1
        if c == bs:
            yield X,y
            c = 0

        
        
        
if __name__=='__main__':
        
        
    gen = load_input(bs=4)
    
    for n,(x,y) in enumerate(gen):
        cv2.imshow('orig', y[n])
        plt.imshow(y[0,:,:,0])
