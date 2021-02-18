import numpy as np
import cv2
import os
import pathlib

path=pathlib.Path().absolute()

def main():
    images=read_images(path)
    for img in images:
        cv2.imshow('img',img)
        cv2.waitKey()




def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(filename,0)
        if img is not None:
            images.append(img)
        else:
            print(filename)
    return images
if __name__=='__main__':
    main()