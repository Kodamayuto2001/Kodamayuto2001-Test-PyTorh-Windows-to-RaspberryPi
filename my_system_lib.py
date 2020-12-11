import os 
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)
        
    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

# class KaoSet:
#     cap_channel         = 0
#     WINDOW_WIDTH        = 1920
#     WINDOW_HEIGHT       = 1080
#     FRAME_WIDTH         = 500
#     FRAME_HEIGHT        = 500
#     color               = (255,0,255)

#     def show(self):
#         cap = cv2.VideoCapture(self.cap_channel)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
#         x   = 100
#         y   = 100
#         while True:
#             success,img = cap.read()
#             cv2.rectangle(img,(x,y),(x+self.FRAME_WIDTH,y+self.FRAME_HEIGHT),self.color,thickness=10)
#             cv2.imshow("Image",img)
#             H,W,C = img.shape
#             x = int((W - self.FRAME_WIDTH)/2)
#             y = int((H - self.FRAME_HEIGHT)/2)
#             cv2.waitKey(100)

class Suiron:
    CAP_CHANNEL         = 0
    WINDOW_WIDTH        = 1920
    WINDOW_HEIGHT       = 1080
    FRAME_WIDTH         = 500
    FRAME_HEIGHT        = 500
    x                   = 100
    y                   = 100
    COLOR               = (255,0,255)
    CASCADEPATH         = "haarcascades/haarcascade_frontalface_default.xml"
    # ---------- 学習の時と同じパラメータでなければならない ---------- #
    inputSize           = 160
    model               = Net(num=6,inputSize=inputSize,Neuron=320)
    PATH                = "models/nn1.pt"
    str_y               = "-------"
    tmp_file_path_w     = "tmp.txt"

    def __init__(self):
        self.cap = cv2.VideoCapture(self.CAP_CHANNEL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
        self.cascade = cv2.CascadeClassifier(self.CASCADEPATH)
        self.model.load_state_dict(torch.load(self.PATH))
        self.model.eval()

    def real_time_haar(self):
        success,img = self.cap.read()
        imgGray     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgResult   = img.copy()
        facerect    = self.cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(200,200))

        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                cv2.rectangle(imgResult,(x,y),(x+w,y+h),self.COLOR,thickness=2)
                imgTrim = img[y:y+h,x:x+w]
                p = self.maesyori_suiron(imgTrim,self.inputSize)
                if p == 0:
                    str_y = "ando   "
                if p == 1:
                    str_y = "higashi"
                if p == 2:
                    str_y = "kataoka"
                if p == 3:
                    str_y = "kodama "
                if p == 4:
                    str_y = "masuda "
                if p == 5:
                    str_y = "suetomo"

                cv2.putText(imgResult, str_y, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2.0,self.COLOR,thickness=2)
                cv2.imshow("Image",imgResult)
                cv2.waitKey(100)

                return str_y
        else:
            str_y = "-------"
            cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
            cv2.imshow("Image",img)
            H,W,C = img.shape
            self.x = int((W - self.FRAME_WIDTH)/2)
            self.y = int((H - self.FRAME_HEIGHT)/2)
            cv2.waitKey(100)
            return str_y
            
    
    def maesyori_suiron(self,imgCV,imgSize):
        # チャンネル数を１
        imgGray = cv2.cvtColor(imgCV,cv2.COLOR_BGR2GRAY)
        
        #リサイズ
        img = cv2.resize(imgGray,(imgSize,imgSize))

        # リシェイプ
        img = np.reshape(img,(1,imgSize,imgSize))

        # transpose h,c,w
        img = np.transpose(img,(1,2,0))

        # ToTensor 正規化される
        img = img.astype(np.uint8)
        mInput = transforms.ToTensor()(img)  

        #推論
        #print(mInput.size())
        output = self.model(mInput[0])

        #予測値
        p = self.model.forward(mInput)

        #予測値出力
        # print(p)
        # print(p.argmax())
        # print(type(p))

        # 戻り値は予測値
        return p.argmax()
