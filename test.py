import my_system_lib as ml
import sys
import cv2

if __name__ == "__main__":
    k = ml.Suiron()
    while True:
        s = k.real_time_haar()
        sys.stdout.write("\r {}さんです。".format(s))
    