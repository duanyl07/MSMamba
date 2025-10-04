import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from collections import defaultdict
from numbers import Number
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        self.dropout1 = nn.Dropout2d(0.3) #0.3\0.1
        self.dropout2 = nn.Dropout2d(0.1)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.dropout1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        out = self.dropout2(out)
        return out

class SSMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SSMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()     
            )
        

    def forward(self,x_graph,x): #[1，2074，dim]
            xg_flat = self.mamba(x_graph) #[1,2074,dim]
            x_recon = x.permute(0, 2, 1).contiguous()
            xg_recon = xg_flat.permute(0, 2, 1).contiguous()
            if self.use_proj:
                x_recon = self.proj(x_recon)
                x_recon = x_recon.permute(0, 2, 1).contiguous()
                xg_recon = self.proj(xg_recon)
                xg_recon = xg_recon.permute(0, 2, 1).contiguous()
            if self.use_residual:
                return xg_recon + x #[1,2074,dim]        #之前是xg_recon + x
            else:
                return x_recon
        
class MSMamba(nn.Module):
    def __init__(self, c1,c2,height: int, width: int, channel: int, class_count: int, dim,flag,Q, nlist,model='normal'):
        super(MSMamba, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channel
        self.dim = dim
        self.Q = Q[0]
        self.Q1 = Q[1]
        # self.A = A[0]
        # self.A1 = A[1]
        self.nlist = nlist
        self.model=model
        self.norm_col_Q = F.normalize(self.Q, p=1, dim=0, eps=1e-9)  # p=1 表示 L1 归一化，dim=0 表示按列归一化
        self.norm_col_Q1 = F.normalize(self.Q1, p=1, dim=0, eps=1e-9)
        self.CNN=flag['CNN']
        self.CE=flag['CE']
        self.GM=flag['GM']
        self.GM1=flag['GM1']
        self.GM2=flag['GM2']
        layers_count=2
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, dim, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(dim),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(dim, dim, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.CNN_Branch.add_module('CNN_Branch'+str(i),SSConv(dim, dim,kernel_size=c1))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(dim, dim, kernel_size=c2))
        
 
        # Softmax layer
        if(self.CNN) and (self.GM or self.GM1 or self.GM2):
            self.Softmax_linear =nn.Sequential(nn.Linear(dim+dim, self.class_count))
        else:
            self.Softmax_linear =nn.Sequential(nn.Linear(dim, self.class_count))

        self.Softmax_linear1 =nn.Sequential(nn.Linear(dim+dim, dim))
        
        # self.GraM1 = SSMamba(dim,use_residual=True,group_num=1)
        self.Mamba = SSMamba(dim,use_residual=True,group_num=1)
        self.use_att=True
        if self.use_att:
            self.weights = nn.Parameter(torch.full((2, dim), 0.5))
            self.softmax = nn.Softmax(dim=0)
            self.weights1 = nn.Parameter(torch.full((2, dim), 0.5))

    def MambaModule(self,nlist,H,Q):
        order = nlist[0]
        reversed_sort=nlist[1]
        HM = H[order]
        HM = HM.unsqueeze(0)
        GraM_result = self.Mamba(HM,HM)
        GraM_result = torch.squeeze(GraM_result, 0)
        GraM_result = GraM_result[reversed_sort,:]
        GraM_result = torch.matmul(Q, GraM_result )
        return GraM_result
    
    def MMamba(self,clean_x_flatten,CNN_result,nlist,norm_col_Q,Q):
        #no-CE
        if self.CE==False:
            superpixels_flatten=torch.mm(norm_col_Q.t(), clean_x_flatten).contiguous()
            H2 = superpixels_flatten
        else:
            clean_x_flatten1=torch.cat([clean_x_flatten,CNN_result],dim=-1).contiguous()
            superpixels_flatten1 = torch.mm(norm_col_Q.t(), clean_x_flatten1)
            H2 = self.Softmax_linear1(superpixels_flatten1)
        
        GraM_result=self.MambaModule( nlist[0],H2,Q) #正向
        GraM_result_r=self.MambaModule( nlist[1],H2,Q) #逆向
    
        weights = self.softmax(self.weights1)
        GraM_result= GraM_result * weights[0] + GraM_result_r * weights[1]  #双向融合
        return  GraM_result
        # return  GraM_result_r

    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape
        # x_flatten = x.reshape([h * w, -1])
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0)) #将光谱特征103变为了dim
        clean_x =torch.squeeze(noise, 0).permute([1, 2, 0]).contiguous()
        hx = clean_x  #[610,340,dim]      Q.t[2074,207400]  clean_x_flatten[207400,dim] 

        #CNN branch
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))# spectral-spatial convolution [1,64,610,340]
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1]) #[207400,64]
        
        clean_x_flatten=clean_x.reshape([h * w, -1])

        #Mamba-Global
        if self.GM1:
            GraM_result = self.MMamba(clean_x_flatten,CNN_result,self.nlist[0],self.norm_col_Q,self.Q)
            if self.GM2:
                GraM_result1 = self.MMamba(clean_x_flatten,CNN_result,self.nlist[1],self.norm_col_Q1,self.Q1)
                weights = self.softmax(self.weights)
                GraM_result= GraM_result * weights[0] + GraM_result1* weights[1]
        elif self.GM2:
                GraM_result = self.MMamba(clean_x_flatten,CNN_result,self.nlist[1],self.norm_col_Q1,self.Q1)
        
        if self.CNN and (self.GM or self.GM1):
            Y = torch.cat([GraM_result,CNN_result],dim=-1)
        elif self.CNN:
            Y = CNN_result
        else :
            Y = GraM_result
            
        y1=Y
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1) #.mean(0)
        Y = Y + 10e-30
        
        return Y,y1