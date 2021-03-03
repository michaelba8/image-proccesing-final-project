import numpy as np
import cv2
import os
import pathlib
import pandas as pd

path=pathlib.Path().absolute()



def main():
    images=read_images(path)
    for index,img in enumerate(images):
        m,n=img.shape
        color_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cut_img = cv2.bilateralFilter(img, 11, 150, 150)
        cut_img = cv2.Canny(cut_img, 50, 120)
        points=find_all_points(cut_img)
        point_7=find_7(cut_img,index,color_img)
        for i in range(points.shape[0]):
            color_img=cv2.circle(color_img,(points[i,0],points[i,1]),3,(0,0,255),thickness=2)
        color_img=cv2.circle(color_img,(point_7[0],point_7[1]),3,(255,0,0),thickness=2)

        point_6,point_8=find_6_and_8(points,point_7)
        color_img = cv2.circle(color_img, (point_6[0], point_6[1]), 3, (255, 0, 0), thickness=2)
        color_img = cv2.circle(color_img, (point_8[0], point_8[1]), 3, (255, 0, 0), thickness=2)
        

        point_5,point_9=find_5_and_9(points,point_6,point_8)
        color_img = cv2.circle(color_img, (point_5[0], point_5[1]), 3, (255, 0, 0), thickness=2)
        color_img = cv2.circle(color_img, (point_9[0], point_9[1]), 3, (255, 0, 0), thickness=2)
        point_1=find_1(points,point_8)
        color_img = cv2.circle(color_img, (point_1[0], point_1[1]), 3, (255, 0, 0), thickness=2)
        point_2 = find_2(points, point_1)
        color_img = cv2.circle(color_img, (point_2[0], point_2[1]), 3, (255, 0, 0), thickness=2)
        point_3 = find_3(points, point_2)
        color_img = cv2.circle(color_img, (point_3[0], point_3[1]), 3, (255, 0, 0), thickness=2)
        point_4 = find_4(points, point_5)
        color_img = cv2.circle(color_img, (point_4[0], point_4[1]), 3, (255, 0, 0), thickness=2)


        dest_points=[point_1,point_2,point_3,point_4,point_5,point_6,point_7,point_8,point_9]
        color_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        color_img=draw_shape(color_img,dest_points)
        cv2.imshow('Hand shape: ',color_img)
        cv2.waitKey()
        df=pd.DataFrame(dest_points,columns={'x','y'})
        file_path = 'part2_image' + str(index) + '.xlsx'
        df.to_excel(file_path, index=False)

        continue






def find_all_points(cut_img):
    dst = cv2.goodFeaturesToTrack(cut_img, 200, 0.0001, 10)
    dst = np.int0(dst[:, 0])
    return dst

def find_7(cut_img,index,img,toShow=False):
    m,n=cut_img.shape
    dst = cv2.goodFeaturesToTrack(cut_img, 60, 0.005, 20)
    dst = np.int0(dst[:, 0])
    points=dst
    for i in range(points.shape[0]):
        color_img = cv2.circle(img, (points[i, 0], points[i, 1]), 3, (0, 0, 255), thickness=2)
    file_name='part1_image'+str(index)+'.xlsx'
    df=pd.read_excel(file_name)
    temp=df.to_numpy()
    fingers=np.zeros(temp.shape)
    for i in range(temp.shape[0]):
        min=1000000
        index=0
        for j in range(temp.shape[0]):
            if(temp[j,1]<min):
                min=temp[j,1]
                index=j
        fingers[i]=temp[index]
        temp[index,1]=10000

    fingers=np.int16(fingers)
    isWork=True
    for i in range(fingers.shape[0]-1):
        if(fingers[i,0]==fingers[i+1,0] and fingers[i,1]==fingers[i+1,1]):
            isWork=False
    if(not isWork):
        dst = cv2.goodFeaturesToTrack(cut_img, 25, 0.005, 20)
        dst = np.int0(dst[:, 0])
        point=find_closest_point(dst,(n/2,m*2/3))
        print('middle')
        return point

    m,n=cut_img.shape
    c1=0.4
    c2=0.6
    x=fingers[0,0]
    y=fingers[2,1]*c1+fingers[3,1]*c2
    point=(x,y)
    point=line(fingers[3],9,163)
    t=point
    color_img = cv2.circle(color_img, (point[ 0], point[ 1]), 3, (0, 255,0 ), thickness=2)
    print(point[0],m-point[1])
    point=find_closest_point(dst,point)
    if(distance(t,point)>20):
        point=t
    color_img = cv2.circle(color_img, (point[0], point[1]), 3, (255, 0, 0), thickness=2)
    print(point[0],m-point[1])
    if(toShow):
        cv2.imshow('sd',color_img)
        cv2.waitKey()
    return point

def find_6_and_8(points,point7):
    point_6=line(point7,65,95)
    t=point_6
    point_6=find_closest_point(points,point_6)
    if(distance(t,point_6)>40):
        point_6=t

    point_8=line(point7,-60,110)
    t=point_8
    point_8 = find_closest_point(points, point_8)
    if (distance(t, point_8) > 40):
        point_8 = t

    return point_6,point_8

def find_5_and_9(points,point_6,point_8):
    point_5=line(point_6,0,47)
    t=point_5
    point_5=find_closest_point(points,point_5)
    if(distance(t,point_5)>40):
        point_5=t

    point_9=line(point_8,0,40)
    t=point_9
    point_9=find_closest_point(points,point_9)
    if (distance(t, point_9) > 40):
        point_9 = t
    return point_5,point_9


def find_3(points,point_2):
    point_3 = line(point_2, 115, 105)
    t=point_3
    point_3 = find_closest_point(points, point_3)
    if (distance(t, point_3) > 70):
        point_3 = t
    return point_3


def find_1(points,point_9):
    point_1=line(point_9,12,110)
    t=point_1
    point_1=find_closest_point(points,point_1)
    if (distance(t, point_1) > 40):
        point_1 = t
    return point_1

def find_2(points,point_1):
    point_2=line(point_1,75,120)
    t=point_2
    point_2=find_closest_point(points,point_2)
    if (distance(t, point_2) > 100):
        point_2 = t
    return point_2


def find_4(points,point_5):
    point_4 = line(point_5, 10, 22)
    t=point_4
    point_4 = find_closest_point(points, point_4)
    if (distance(t, point_4) > 40):
        point_4 = t
    return point_4


def line(point,degree,distance):
    alpha=np.deg2rad(degree)
    x=int(point[0]+distance*np.cos(alpha))
    y=int(point[1]-distance*np.sin(alpha))
    return (x,y)

def find_closest_point(points,point1):
    if(points.shape[0]==1):
        return points[0]
    min_dist=100000
    min_idx=-1
    for i in range(points.shape[0]):
        d=distance(point1,points[i])
        if(d<min_dist):
            min_dist=d
            min_idx=i
    return points[min_idx]


def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 +(point1[1]-point2[1])**2)

def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(filename,0)
        if img is not None:
            images.append(img)
    return images

def draw_shape(img,points,to_draw=False):
    img=np.copy(img)
    points=np.copy(points)
    first=points[0]
    this=0
    while(points.shape[0]>1):
        min_dis=100000
        index=0
        this_point = points[this]
        points = np.delete(points, this, 0)
        for j in range (points.shape[0]):
            dist=distance(this_point,points[j])
            if(dist<min_dis):
                min_dis=dist
                index=j
        cv2.line(img,(this_point[0],this_point[1]),(points[index][0],points[index][1]),(0,0 ,255),3)
        this = index
        if(to_draw):
            cv2.imshow('s', img)
            cv2.waitKey()
    cv2.line(img, (points[0][0],points[0][1]), (first[0],first[1]),(0,0 ,255), 3)
    if(to_draw):
        cv2.imshow('s',img)
        cv2.waitKey()
    return img









if __name__=='__main__':
    main()