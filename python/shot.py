import cv2
import sys
import numpy as np
import datetime

def shot(shot_num=20, image_dir="./images", name=None):
    if name is None:
        name = datetime.datetime.now()
    cap = cv2.VideoCapture(0)
    i = 0
    print("Press C to take a photo or Q to quit.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if shot_num == 0:
                cv2.imwrite(f"{image_dir}/{name}.png",frame)
                break
            else:
                cv2.imwrite(f"{image_dir}/{name}_{i}.png",frame)
                print(f"{i+1}/{shot_num} done.")
            i += 1
        elif key == ord('q'):
            break
        if i>=shot_num and shot_num!=0:
            break

if __name__ == "__main__":
    if len(sys.argv) == 3:
        shot(shot_num=int(sys.argv[1]), name=sys.argv[2])
    elif len(sys.argv) == 2:
        shot(shot_num=0, name=sys.argv[1])
    else:
        shot()