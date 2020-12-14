import numpy as np 
import random 
import cv2 

class DataAugmentation:
    MIN_TABLE       = 50
    MAX_TABLE       = 205 
    DIFF_TABLE      = MAX_TABLE - MIN_TABLE
    GANMA1          = 0.75 
    GANMA2          = 1.5 
    AVERAGE_SQUARE  = (10,10)
    MEAN            = 0
    SIGMA           = 15
    def __init__(self):
        pass 

    def hi_contrast(self,img):
        LUT_HC  = np.arange(256,dtype="uint8")
        for i in range(0,self.MIN_TABLE):
            LUT_HC[i] = 0
        for i in range(self.MIN_TABLE,self.MAX_TABLE):
            LUT_HC[i] = 255*(i-self.MIN_TABLE)/self.DIFF_TABLE
        for i in range(self.MAX_TABLE,255):
            LUT_HC[i] = 255
        hi_cont_img = cv2.LUT(img,LUT_HC)
        
        return hi_cont_img
        
    def lo_contrast(self,img):
        LUT_LC  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_LC[i] = self.MIN_TABLE + i * (self.DIFF_TABLE) / 255
        lo_cont_img = cv2.LUT(img,LUT_LC)

        return lo_cont_img

    def ganma_1(self,img):
        LUT_G1  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_G1[i] = 255 * pow(float(i)/255,1.0/self.GANMA1)
        ganma1_img = cv2.LUT(img,LUT_G1)

        return ganma1_img

    def ganma_2(self,img):
        LUT_G2  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_G2[i] = 255 * pow(float(i)/255,1.0/self.GANMA2)
        ganma2_img = cv2.LUT(img,LUT_G2)

        return ganma2_img

    def blur(self,img):
        return cv2.blur(img,self.AVERAGE_SQUARE)

    def gauss(self,img):
        row,col,ch = img.shape 
        gauss = np.random.normal(self.MEAN,self.SIGMA,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss_img = img + gauss

        return gauss_img

    def sp_noise(self,img,prob):
        sp_noise_img  = np.zeros(img.shape,np.uint8)
        thres   = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255 
                else:
                    output[i][j] = img[i][j]
        
        return sp_noise_img