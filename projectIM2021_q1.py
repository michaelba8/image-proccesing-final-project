import numpy as np
import cv2
import os
import pathlib
import projectIM2021_q2 as q2
import pandas as pd

path=pathlib.Path().absolute()

def main():
    images=read_images(path)
    for i,img in enumerate(images):
        temp = cv2.bilateralFilter(img,21,150,150)
        temp = cv2.Canny(img, 80, 200)
        cut_img=np.copy(temp)
        t1 = int(temp.shape[0] * 54/ 200)
        t2=int((21*temp.shape[1])/30)
        t3=int(t2*18/30)
        t4=int(temp.shape[0]/2)
        t5=int(temp.shape[0]*48/60)
        t6=int(temp.shape[1]*13/20)
        cut_img[t1:,t3:] = 0
        cut_img[:,t2:]=0
        cut_img[t4:t5,t6:]=0

        img,fingers = fingers_detetction(img, temp,cut_img)
        df = pd.DataFrame(fingers,columns={'x','y'})
        file_path='part1_image'+str(i)+'.xlsx'
        df.to_excel(file_path, index=False)
        cv2.imshow('Fingers: ',img)
        cv2.waitKey()

def fingers_detetction(img,cut_img,cut_img2):
    """detects excatly 5 fingers and marks them with red dot"""
    circles = cv2.HoughCircles(cut_img,cv2.HOUGH_GRADIENT,20,40,
                                param1=50,param2=10,minRadius=7,maxRadius=27)
    cimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))
    circles=remove_the_rest(cut_img,circles,cut_img2)
    for point in circles:
        cimg=cv2.circle(cimg,(point[0],point[1]),3,(0,0,255),2)
    circles=remove_the_rest_2(img,circles)
    circles=remove_the_rest_2(cut_img2,circles,True)
    circles=remove_the_rest_3(img,circles)
    cimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(circles.shape[0]):
        point=circles[i]
        cimg=cv2.circle(cimg,(point[0],point[1]),3,(0,0,255),2)


    return cimg,circles

def read_images(folder):
    """Reading all the image from folder"""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(filename,0)
        if img is not None:
            images.append(img)

    return images

def remove_the_rest_2(img,circles,toCut=False):
    """removing extra points by moving on the image from the point to left and see if meeting a line
        when #toCut==False the function will get an input of image after Canny and deleting unnecessary parts of the image"""
    if(toCut==False):
        cut_img=cv2.Canny(img,90,175)
    cut_img=img
    dlt=[]
    max_len=35
    for i in range(len(circles)):
        length=0
        x=circles[i][0]
        y=circles[i][1]-2
        while(length<max_len and x>0):
            if(cut_img[y,x]!=0):
                break
            x-=1
            length+=1
        if(length==max_len or x==0):
            dlt.append(i)
    circles=np.delete(circles,dlt,0)
    return circles

def remove_the_rest_3(img,circles):
    """removing extra circles that left, by drawing lines from the thumb to the other fingers"""
    thumb=0
    min=10000
    if(circles.shape[0]==5):
        return circles
    fingers=np.zeros((5,2))
    for i in range(circles.shape[0]):
        y=circles[i,1]
        if(y<min):
            min=y
            thumb=i
    fingers[0,:]=circles[thumb,:]
    finger2=q2.line(fingers[0],-165,210)
    finger2=q2.find_closest_point(circles,finger2)
    finger3=q2.line(finger2,-90,50)
    finger3=q2.find_closest_point(circles,finger3)
    finger4 = q2.line(finger3, -75, 50)
    finger4 = q2.find_closest_point(circles, finger4)
    finger5=q2.line(finger4,-45,75)
    finger5=q2.find_closest_point(circles,finger5)
    fingers[1:]=finger2
    fingers[2:] = finger3
    fingers[3:] = finger4
    fingers[4:] = finger5
    fingers=np.int16(fingers)
    return fingers

def remove_the_rest(img,circles,cut_img):
    """removing extra points from hough_circles by detetcting areas that doesnt have much colors after Canny"""
    m,n=img.shape
    circles=np.reshape(circles,(circles.shape[1],circles.shape[2]))
    dlt=[]
    for i,point in enumerate(circles):
        if(point[0]>=n or point[1]>=m):
            dlt.append(i)
    circles=np.delete(circles,dlt,0)
    filter=np.ones((25,25))/625
    cut_img=cv2.filter2D(cut_img,-1,filter)
    l=[]
    for i,point in enumerate(circles):
        l.append((cut_img[point[1],point[0]],i))
    l.sort(key=lambda tup: tup[0],reverse=True)
    res=[]
    for i in range(12):
        res.append(circles[l[i][1]])
    res=np.array(res)
    #for point in res:
    #    img=cv2.circle(img,(point[0],point[1]),3,255,2)
    res=np.delete(res,2,1)
    return res


if __name__=='__main__':
    main()