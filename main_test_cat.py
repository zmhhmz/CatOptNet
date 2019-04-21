
# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set68'], help='directory of test dataset')
    parser.add_argument('--sigma', default=40, type=float, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'cat_peak40_noup_lam0.1_epo2'), help='directory of the model')
    parser.add_argument('--model_name', default='model_002.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

    
class CNNBlock0(nn.Module): #momentum
    def __init__(self, image_channels, n_channels, level):
        super(CNNBlock0, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=image_channels,
                                 out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(level-2):
            layers2.append(nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers2.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.9))
            layers2.append(nn.ReLU(inplace=True))
#        layers2.append(nn.Conv2d(in_channels=n_channels,
#                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x):
        out = self.cnnblock(x)
        return out

class CNNBlock1(nn.Module): #momentum
    def __init__(self, image_channels, n_channels):
        super(CNNBlock1, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=n_channels,
                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x, x0):
        out = self.cnnblock(x)
        return out+x0

class CNNBlock2(nn.Module): #momentum
    def __init__(self, image_channels, n_channels, level):
        super(CNNBlock2, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=image_channels+n_channels,
                                 out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(level-2):
            layers2.append(nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers2.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.9))
            layers2.append(nn.ReLU(inplace=True))
#        layers2.append(nn.Conv2d(in_channels=n_channels,
#                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x,x0):
        x1=torch.cat((x0,x),1)
        out = self.cnnblock(x1)
        return out


class PADMM(nn.Module):
    def __init__(self,tao=1,eta=0.1,decay=0.9, level=6, subnet=5, n_channels=64, image_channels=1):
        super(PADMM, self).__init__()
        self.tao=tao
        self.level = level
        self.eta=eta
        self.decay = decay
        self.proxNet0 = CNNBlock0(image_channels,n_channels,subnet)
        self.proxNet1 = CNNBlock1(image_channels,n_channels)
        self.proxNet2 = CNNBlock2(image_channels,n_channels,subnet)
        #self.proxNet_final = CNNBlock1(image_channels,n_channels,final_subnet)
        self._initialize_weights()

    def updateX(self,V,Y,tao):
        return (V-tao+torch.sqrt_((V-tao)**2+4*tao*Y)).div_(2)

    def forward(self, Y):
        X = Y
        X_p= Y
        listV = []
        tao = self.tao
        temp = self.proxNet0(X)
        for i in range(self.level-1):
            V = self.proxNet1(temp,X)
            listV.append(V)
            V=V+self.eta*(X-X_p)
            X_p=X
            X = self.updateX(V,Y,tao)
            X=V
            tao *=self.decay
            temp = self.proxNet2(X,temp)
        V = self.proxNet1(temp,X)
        return listV, V
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               #init.kaiming_normal_(m.weight,nonlinearity='relu')
                init.orthogonal_(m.weight)
               #print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1/100)
                init.constant_(m.bias, 0)


def imshow(X):
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X.squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    args = parse_args()
#    sigma = args.sigma
#    lam = args.lam

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())

    model.eval()  # evaluation mode
#    model.train()

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []
        k=0
        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                np.random.seed(seed=0)  # for reproducibility
                #y = np.random.poisson(255.0*x*args.sigma)/(255.0*args.sigma)
                x255=x*255.0
                Q=np.amax(x255,axis=(0,1))/args.sigma
                im_lam=x255/Q#np.broadcast_to(Q[:,None,None],x255.shape)
                im_lam[im_lam==0]=np.amin(im_lam[im_lam>0])   
                y = Q*np.random.poisson(im_lam).astype('float32')/255.0
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)[1]  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    if k<10:
                        show(np.hstack((y, x_, y-x_)))  # show the image
                    k=k+1
                    save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))








