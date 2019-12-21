import cv2
from model.show_img_code import  get_img

im = get_img()
cv2.imshow("test", im)
cv2.waitKey(500)