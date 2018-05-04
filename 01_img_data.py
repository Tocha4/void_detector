import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

img_names = os.listdir('../Mikro_markiert')

img = cv2.imread(os.path.join('../Mikro_markiert', img_names[3]))

ret, new_img = cv2.threshold(img[:,:,2], 250, 255, cv2.THRESH_BINARY)

cv2.imshow('mask', new_img)
cv2.imshow('original', img)