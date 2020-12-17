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
    WINDOW_WIDTH        = 720
    WINDOW_HEIGHT       = 480
    FRAME_WIDTH         = 300
    FRAME_HEIGHT        = 300
    x                   = 100
    y                   = 100
    COLOR               = (255,0,0)
    CASCADEPATH         = "haarcascades/haarcascade_frontalface_default.xml"
    MOJI_OOKISA         = 1.0
    # ---------- 学習の時と同じパラメータでなければならない ---------- #
    inputSize           = 160
    model               = Net(num=6,inputSize=inputSize,Neuron=320)
    PATH                = "models/nn1.pt"
    BODY_TEMP           = 36.5
    BODY_TEMP_SAFE      = (255,0,0)
    BODY_TEMP_OUT       = (255,0,255)
    DELAY_MSEC          = 1

    CNT_ANDO            =   0
    CNT_HIGASHI         =   0
    CNT_KATAOKA         =   0
    CNT_KODAMA          =   0
    CNT_MASUDA          =   0
    CNT_SUETOMO         =   0
    CNT                 =   0
    CNT_MAX             =   100
    PROGRESS_BAR_LEN    =   100

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

        H,W,C = img.shape
        self.x = int((W - self.FRAME_WIDTH)/2)
        self.y = int((H - self.FRAME_HEIGHT)/2)

        #   もし体温が37.0度以上の時は赤、未満は青
        if self.BODY_TEMP >= 37.0:
            self.COLOR  =   self.BODY_TEMP_OUT
        else:
            self.COLOR  =   self.BODY_TEMP_SAFE

        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                cv2.rectangle(imgResult,(x,y),(x+w,y+h),self.COLOR,thickness=2)
                imgTrim = img[y:y+h,x:x+w]
                str_y,percent,ld = self.maesyori_suiron(imgTrim,self.inputSize)

                cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
                cv2.putText(img, str_y+" "+str(percent)+"%", (40, 40), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                cv2.putText(img,"Body TEMP",(40,40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                cv2.putText(img,str(self.BODY_TEMP),(40,40*3),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)


                cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                        int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+self.PROGRESS_BAR_LEN,  int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (204,204,204),
                        15
                )

                #   もし、ldが  "-------"ではないとき
                if ld != "-------":
                    print("ok")
                else:
                    cv2.putText(img,"Please",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,"wait.",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                                                    int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+(int(self.PROGRESS_BAR_LEN/self.CNT_MAX))*self.CNT, int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        self.COLOR,
                        15
                    )

                cv2.imshow("Image",img)
                cv2.waitKey(self.DELAY_MSEC)

                return ld
        else:
            #   もし顔が認識できていなかったらCNTをリセットする
            self.CNT    =   0
            str_y = "-------"
            cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
            cv2.putText(img, "Set Face", (40*2, 40*2), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA*2,self.COLOR,thickness=4)
            cv2.imshow("Image",img)
            cv2.waitKey(self.DELAY_MSEC)
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
        # p0 = self.model.forward(mInput)
        p1 = self.model.forward(mInput).exp()

        #予測値のパーセンテージ
        # m = torch.nn.Softmax(dim=1)
        # x0 = m(p0)
        # x0 = x0.to('cpu').detach().numpy().copy() 
        # x0 = x0[0]
        x1 = p1.to('cpu').detach().numpy().copy() 
        x1 = x1[0]
        # すべての中で最も大きい値
        p1 = p1.argmax()
        percent = 0

        if p1 == 0:
            self.CNT_ANDO       +=  1
            str_y = "ando   "
            percent = x1[0]*100
            # print(percent)
            # print(x1[0]*100)
        if p1 == 1:
            self.CNT_HIGASHI    +=  1
            str_y = "higashi"
            percent = x1[1]*100
            # print(percent)
            # print(x1[1]*100)
        if p1 == 2:
            self.CNT_KATAOKA    +=  1
            str_y = "kataoka"
            percent = x1[2]*100
            # print(percent)
            # print(x1[2]*100)
        if p1 == 3:
            self.CNT_KODAMA     +=  1
            str_y = "kodama "
            percent = x1[3]*100
            # print(percent)
            # print(x1[3]*100)
        if p1 == 4:
            self.CNT_MASUDA     +=  1
            str_y = "masuda "
            percent = x1[4]*100
            # print(percent)
            # print(x1[4]*100)
        if p1 == 5:
            self.CNT_SUETOMO    +=  1
            str_y = "suetomo"
            percent = x1[5]*100
            # print(percent)
            # print(x1[5]*100)

        cnt_list    =   [
            self.CNT_ANDO,
            self.CNT_HIGASHI,
            self.CNT_KATAOKA,
            self.CNT_KODAMA,
            self.CNT_MASUDA,
            self.CNT_SUETOMO
        ]
        self.CNT    +=  1

        s = "-------"
        #   リセット
        if self.CNT == self.CNT_MAX:
            max_value   =   max(cnt_list)
            max_index   =   cnt_list.index(max_value)
            if max_index == 0:
                s = "ando   "
            if max_index == 1:
                s = "higashi"
            if max_index == 2:
                s = "kataoka"
            if max_index == 3:
                s = "kodama "
            if max_index == 4:
                s = "masuda "
            if max_index == 5:
                s = "suetomo"
            self.CNT        = 0
            self.CNT_ANDO   = 0
            self.CNT_HIGASHI= 0
            self.CNT_KATAOKA= 0
            self.CNT_KODAMA = 0
            self.CNT_MASUDA = 0
            self.CNT_SUETOMO= 0

        

        # 戻り値は予測値とパーセンテージ,確実な値
        return str_y,percent,s

    def imshow(self,path):
        img = cv2.imread(path)
        cv2.imshow("Thermo sensor",img)
        cv2.waitKey(self.DELAY_MSEC)
        # cv2.moveWindow("Thermo sensor",self.WINDOW_WIDTH,int(self.WINDOW_HEIGHT/2))