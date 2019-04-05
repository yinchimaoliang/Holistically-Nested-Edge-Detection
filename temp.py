import cv2 as cv




PATH = "./Dataset/images/test/232076.jpg"

img = cv.imread(PATH,0)
for i in img:
    for j in i:
        if j < 0:
            print("test")