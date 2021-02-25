import numpy as np
import cv2
import os
import pathlib

path=pathlib.Path().absolute()

def main():
    images=read_images(path)
    #part_one(images)
    part_two(images)


def part_two(images):

    for img in images:
        m,n=img.shape
        color_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cut_img = cv2.bilateralFilter(img, 11, 150, 150)
        cut_img = cv2.Canny(cut_img, 50, 120)
        dest_points=[]
        #cv2.imshow('sd',np.hstack((cut_img,img)))
        #cv2.waitKey()
        points=find_all_points(cut_img)
        point_7=find_7(cut_img)
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
        cv2.imshow('sds', color_img)
        cv2.waitKey()

        dest_points=[point_1,point_2,point_3,point_4,point_5,point_6,point_7,point_8]
        color_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        color_img=draw_shape(color_img,dest_points)
        cv2.imshow('sds',color_img)
        cv2.waitKey()
        continue


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




def find_all_points(cut_img):
    dst = cv2.goodFeaturesToTrack(cut_img, 200, 0.0001, 10)
    dst = np.int0(dst[:, 0])
    return dst

def find_7(cut_img):
    dst = cv2.goodFeaturesToTrack(cut_img, 25, 0.005, 20)
    dst = np.int0(dst[:, 0])
    m,n=cut_img.shape
    point=find_closest_point(dst,(n*1/2,m*2/3))
    return point

def find_6_and_8(points,point7):
    point_8=line(point7,60,110)
    point_8=find_closest_point(points,point_8)
    point_6=line(point7,-78,95)
    point_6 = find_closest_point(points, point_6)
    return point_6,point_8

def find_5_and_9(points,point_6,point_8):
    point_5=line(point_6,0,47)
    point_5=find_closest_point(points,point_5)
    point_9=line(point_8,0,40)
    point_9=find_closest_point(points,point_9)
    return point_5,point_9
def find_3(points,point_2):
    point_1 = line(point_2, 210, 105)
    point_1 = find_closest_point(points, point_1)
    return point_1


def find_1(points,point_9):
    point_1=line(point_9,-10,100)
    point_1=find_closest_point(points,point_1)
    return point_1

def find_2(points,point_1):
    point_2=line(point_1,-75,120)
    point_2=find_closest_point(points,point_2)
    return point_2


def find_4(points,point_5):
    point_4 = line(point_5, -10, 22)
    point_4 = find_closest_point(points, point_4)
    return point_4


def line(point,degree,distance):
    alpha=np.deg2rad(degree)
    x=int(point[0]+distance*np.cos(alpha))
    y=int(point[1]+distance*np.sin(alpha))
    return (x,y)

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

def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 +(point1[1]-point2[1])**2)

def part_one(images):
    """part one of the project, get 4 (or more) hands images and marks the top of the each finger"""
    cut_images=[]
    for img in images:
        #temp = cv2.bilateralFilter(img,9,75,75)
        temp=cv2.Canny(img,84,165)
        #temp=cv2.Canny(temp,25,80)

        t1 = int(temp.shape[0] * 53 / 200)  # ~m*1/5
        t2=int((21*temp.shape[1])/30)   # ~n*2/3
        t3=int(t2*18/30)                # ~n*4/9
        t4=int(temp.shape[0]/2)         # ~m*1/2
        t5=int(temp.shape[0]*48/60)     # ~m*5/6
        t6=int(temp.shape[1]*1/3)       # ~n*1/3

        temp[t1:,t3:] = 0
        temp[:,t2:]=0
        temp[t4:t5,t6:]=0

        cut_images.append(temp)
        fin=fingers_detetction(img,temp)
        #cv2.imshow('img',np.hstack((temp,fin)))
        cv2.imshow('img',fin)
        cv2.waitKey()

def fingers_detetction(img,cut_img):
    """detects excatly 5 fingers and marks them with red dot"""
    circles = cv2.HoughCircles(cut_img,cv2.HOUGH_GRADIENT,31,42,
                                param1=50,param2=10,minRadius=7,maxRadius=26)
    #circles = cv2.HoughCircles(cut_img, cv2.HOUGH_GRADIENT, 31, 40,
    #                           param1=50, param2=40, minRadius=9, maxRadius=26)
    cimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))
    circles[0,:,1]+=5
    np.delete(circles[0,:,:],5)
    circles=check_validation(cut_img,circles[:,:6,:])
    for i in circles[0,:5]:
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        #cv2.circle(cimg, (i[0], i[1]), 2, (255), 3)
    return cimg

def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(filename,0)
        if img is not None:
            images.append(img)
    return images

def check_validation(img,circles):
    """remove extra points that the algorithm Hough Transform finds"""
    m,n=img.shape
    l=[]
    for i in range(circles.shape[1]):
        point=(circles[0,i,0],circles[0,i,1],circles[0,i,2])
        l.append(point)
    l.sort(key=lambda tup: tup[1])
    first= l.pop(0)
    last= l.pop()
    avg=0
    for x,y,radius in l:
        avg+=x/len(l)
    max_len=0
    max_len_idx=0
    for i in range(len(l)):
        temp=(l[i][0]-avg)**2
        if (temp>max_len):
            max_len=temp
            max_len_idx=i
    del l[max_len_idx]
    l.append(first)
    l.append(last)
    circles=np.uint16([l])
    return circles

if __name__=='__main__':
    main()