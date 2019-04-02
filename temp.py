import cv2 as cv




PATH = "./Dataset/images/test/232076.jpg"

img = cv.imread(PATH,0)
cv.resize(img,(100,100))