import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    h = 960
    w = 1280
    img = cv2.imread("tmp.png")
    img = img[:h, :w]
    print(img.shape)
    for i in range(h):
        if i%50!=0:
            continue
        cv2.line(img, (0,i), (w,i), (0,255,0), 2)
    for i in range(w):
        if i%50!=0:
            continue
        cv2.line(img, (i,0), (i,h), (255,0,0), 2)

    fx = 135
    fy = 135
    cx = 655
    cy = 508
    rad_x = math.atan(w/fx)
    rad_y = math.atan(h/fy)
    map_x = np.zeros((h,w), dtype=np.float32)
    map_y = np.zeros((h,w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            lamb = (1.0 - y/(h/2.0))*rad_y
            phi = (x/(w/2.0) - 1.0)*rad_x
            rec_x = cx + fx*math.tan(phi)/math.cos(lamb)
            rec_y = cy - fy*math.tan(lamb)
            map_x[y,x] = rec_x
            map_y[y,x] = rec_y

    rev_map_x = np.zeros((h,w), dtype=np.float32)
    rev_map_y = np.zeros((h,w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            lamb = math.atan((cy-y)/fy)
            phi = math.atan(math.cos(lamb)*(x-cx)/fx)
            eqrec_x = (1+phi/rad_x)*(w/2.0)
            eqrec_y = (1-lamb/rad_y)*(h/2.0)
            rev_map_x[y,x] = eqrec_x
            rev_map_y[y,x] = eqrec_y

    eqrec = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    rect = cv2.remap(eqrec, rev_map_x, rev_map_y, cv2.INTER_LINEAR)
    cv2.imshow("test", eqrec)
    cv2.imshow("test2", rect)
    cv2.waitKey(0)