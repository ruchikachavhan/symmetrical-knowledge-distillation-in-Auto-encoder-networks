import cv2
import numpy as np
import os

X_test = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/test_X/"
Y_test = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/test_Y/"

X_dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/X/"
Y_dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/Y/"

X_dir_val = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/X_val/"
Y_dir_val = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/Y_val/"

dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/train/"
val_dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/val"


for images in os.listdir(dir):
    image = cv2.imread(dir + images)
    print(image.shape)
    x = image[:, 0:256:, :]
    y = image[:, 256:512, :]
    x_name = X_dir + images
    y_name = Y_dir + images
    cv2.imwrite(x_name, x)
    cv2.imwrite(y_name, y)


for images in os.listdir(val_dir):
    image = cv2.imread(val_dir + images)
    print(image.shape)
    x = image[:, 0:256:, :]
    y = image[:, 256:512, :]
    x_name = X_dir_val + images
    y_name = Y_dir_val + images
    cv2.imwrite(x_name, x)
    cv2.imwrite(y_name, y)


index = 0
for images in os.listdir(X_dir_val):
    if(index<100):
        index+=1
        image = cv2.imread(X_dir_val + images)
        print(image.shape)
        cv2.imwrite(X_test+images, image)

index = 0
for images in os.listdir(Y_dir_val):
    if(index<100):
        index+=1
        image = cv2.imread(Y_dir_val + images)
        print(image.shape)
        cv2.imwrite(Y_test+images, image)
