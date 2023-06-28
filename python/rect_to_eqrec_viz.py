import math
import numpy as np
import cv2

if __name__ == "__main__":

    fs = cv2.FileStorage("./configs/camera_params.yaml", cv2.FILE_STORAGE_READ)
    img_size = fs.getNode("img_size")
    w, h = int(img_size.at(0).real()), int(img_size.at(1).real())

    camera_params = fs.getNode("camera_params")
    K1 = camera_params.getNode('K1').mat()
    D1 = camera_params.getNode('D1').mat()
    R1 = camera_params.getNode('R1').mat()
    P1 = camera_params.getNode('P1').mat()
    fs.release()

    # img = np.full((h,w,3), 255, np.uint8)
    img = cv2.imread("./images/test-1.jpg")[:h,:w]

    for i in range(h):
        if i%40!=0:
            continue
        cv2.line(img, (0,i), (w,i), (0,255,0), 2)
    for i in range(w):
        if i%40!=0:
            continue
        cv2.line(img, (i,0), (i,h), (255,0,0), 2)

    fx = P1[0,0]
    fy = P1[1,1]
    cx = P1[0,2]
    cy = P1[1,2]
    rad_x = math.atan(w/fx)
    rad_y = math.atan(h/fy)

    rect_map_x, rect_map_y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_32FC1)
    
    rect_to_eqrec_map_x = np.zeros((h,w), dtype=np.float32)
    rect_to_eqrec_map_y = np.zeros((h,w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            lamb = (1.0 - y/(h/2.0))*rad_y
            phi = (x/(w/2.0) - 1.0)*rad_x
            rec_x = cx + fx*math.tan(phi)/math.cos(lamb)
            rec_y = cy - fy*math.tan(lamb)
            rect_to_eqrec_map_x[y,x] = rec_x
            rect_to_eqrec_map_y[y,x] = rec_y

    eqrec_map_x = cv2.remap(rect_map_x, rect_to_eqrec_map_x, rect_to_eqrec_map_y, cv2.INTER_LINEAR)
    eqrec_map_y = cv2.remap(rect_map_y, rect_to_eqrec_map_x, rect_to_eqrec_map_y, cv2.INTER_LINEAR)

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

    rect = cv2.remap(img, rect_map_x, rect_map_y, cv2.INTER_LINEAR)
    eqrec_coarse = cv2.remap(rect, rect_to_eqrec_map_x, rect_to_eqrec_map_y, cv2.INTER_LINEAR)
    eqrec = cv2.remap(img, eqrec_map_x, eqrec_map_y, cv2.INTER_LINEAR)

    cv2.imshow("original", img)
    cv2.imwrite("./images/gird.png", img)
    cv2.imshow("rectified", rect)
    cv2.imwrite("./images/grid_rect.png", rect)
    cv2.imshow("equirectangular coarse", eqrec_coarse)
    cv2.imwrite("./images/grid_eqrec_coarse.png", eqrec_coarse)
    cv2.imshow("equirectangular", eqrec)
    cv2.imwrite("./images/grid_eqrec.png", eqrec)
    cv2.waitKey(0)