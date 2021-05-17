# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:29:01 2020

@author: pc
"""

import torch
import torch.nn as nn
from spectral import SpectralNorm
import torch.nn.functional as F

'''多残差模块''' 
class MutiRes_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(in_channels, affine=False),
            nn.LeakyReLU(0.1, inplace=True) )
        
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(in_channels, affine=False),
            nn.LeakyReLU(0.1, inplace=True) )
        
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(out_channels, affine=False) ) 

    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        out = x4 + x3 + x1 + x   
        return out


'''下采样模块'''  
class Downsample_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.1, inplace=True) )

    def forward(self, x): 
        out = self.conv(x)      
        return out
   
'''''''''判别器'''''''''
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Discriminator, self).__init__()
   
        self.conv1 = Downsample_block(in_ch, 64)  
        self.conv2 = Downsample_block(64, 128)  
        self.conv3 = Downsample_block(128, 256) 
        self.conv4 = MutiRes_block(256, 256) 
        self.conv5 = MutiRes_block(256, 256) 
        self.conv6 = MutiRes_block(256, 256) 
        self.conv7 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, out_ch, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.Sigmoid() )          

    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
#        x5 = self.conv5(x4)
#        x6 = self.conv6(x5)
        out = self.conv7(x4)
        return out 
    
class REBNCONV1(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV1,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
    
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV1(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV1(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV1(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV1(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV1(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV1(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV1(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV1(mid_ch*2,out_ch,dirate=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = self.upscore2(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = self.upscore2(hx2d)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

# 深度可分离模块
class REBNCONV(nn.Module):
    def __init__(self,in_ch,out_ch,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch , in_ch*4,3,padding=1*dirate, dilation=1*dirate, bias = False)
        self.bn_s1 = nn.BatchNorm2d(in_ch*4)
        self.relu_s1 = nn.ReLU()
        
        self.conv_s2 = nn.Conv2d(in_ch*4, in_ch*4,3,padding=1*dirate, stride=1, groups=in_ch*4, dilation=1*dirate, bias = False)
        self.bn_s2 = nn.BatchNorm2d(in_ch*4)
        self.relu_s2 = nn.ReLU()
        
        self.conv_s3 = nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1, padding=0, dilation=1*dirate, bias=False)
        self.bn_s3 = nn.BatchNorm2d(out_ch)
        self.relu_s3 = nn.ReLU()

    def forward(self,x):

        hx = x
        hx = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.relu_s2(self.bn_s2(self.conv_s2(hx)))
        xout = self.relu_s3(self.bn_s3(self.conv_s3(xout)))

        return xout

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = self.upscore2(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = self.upscore2(hx2d)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

'''SE注意力模块''' 
class SE_Attn1(nn.Module):
    def __init__(self, channel):
        super(SE_Attn1,self).__init__()
    
        self.squeeze = nn.AdaptiveAvgPool2d(1) 
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel//16),
            nn.PReLU(),
            nn.Linear(channel//16, channel),
            nn.Sigmoid())
        
    def forward(self, x):
        x1 = self.squeeze(x)
        x2 = x1.view(x1.size(0), -1)                 
        x3 = self.excitation(x2)
        out = x3.view(x.size(0), x.size(1), 1, 1)    
        return out 
    

class SA_Attn(nn.Module):
    def __init__(self, channel):
        super().__init__()
    
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel//2),
            nn.PReLU(),
            nn.Conv2d(channel//2, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        
    def forward(self, x):
        out = self.demo(x)
        return out 

class AttentionGate1(nn.Module):
    def __init__(self,channel):
        super(AttentionGate1, self).__init__()
        self.space = SA_Attn(channel)
        self.chanl = SE_Attn1(channel)
    def forward(self, a):
        f_weight = self.space(a)
        c_weight = self.chanl(a)
        att_f = a * f_weight
        att_fc = att_f * c_weight
        out = a + att_fc
        return out

'''三重注意力模块'''      
class TH_Attn(nn.Module):
    def __init__(self, channel,height, no_spatial=False):
        super(TH_Attn, self).__init__()
        self.cw = AttentionGate1(height)
        self.hc = AttentionGate1(height)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate1(channel)
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
          
        
#'''三重注意力模块'''
#class BasicConv(nn.Module):
#    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#        super(BasicConv, self).__init__()
#        self.out_channels = out_planes
#        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#        self.relu = nn.ReLU() if relu else None
#
#    def forward(self, x):
#        x = self.conv(x)
#        if self.bn is not None:
#            x = self.bn(x)
#        if self.relu is not None:
#            x = self.relu(x)
#        return x
#class ZPool(nn.Module):
#    def forward(self, x):
#        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
#
#class AttentionGate(nn.Module):
#    def __init__(self):
#        super(AttentionGate, self).__init__()
#        kernel_size = 3
#        self.compress = ZPool()
#        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#    def forward(self, x):
#        x_compress = self.compress(x)
#        x_out = self.conv(x_compress)
#        scale = torch.sigmoid_(x_out) 
#        return x * scale
#
#class SE_Attn(nn.Module):
#    def __init__(self, no_spatial=False):
#        super(SE_Attn, self).__init__()
#        self.cw = AttentionGate()
#        self.hc = AttentionGate()
#        self.no_spatial=no_spatial
#        if not no_spatial:
#            self.hw = AttentionGate()
#    def forward(self, x):
#        x_perm1 = x.permute(0,2,1,3).contiguous()
#        x_out1 = self.cw(x_perm1)
#        x_out11 = x_out1.permute(0,2,1,3).contiguous()
#        x_perm2 = x.permute(0,3,2,1).contiguous()
#        x_out2 = self.hc(x_perm2)
#        x_out21 = x_out2.permute(0,3,2,1).contiguous()
#        if not self.no_spatial:
#            x_out = self.hw(x)
#            x_out = 1/3 * (x_out + x_out11 + x_out21)
#        else:
#            x_out = 1/2 * (x_out11 + x_out21)
#        return x_out
#    

class Generator(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(Generator,self).__init__()
        
        '''注意力模块2'''
        self.SE_chanl1RS = TH_Attn(16,256)
        self.SE_chanl2RS = TH_Attn(32,128)
        self.SE_chanl3RS = TH_Attn(64,64)
        self.SE_chanl4RS = TH_Attn(128,32)
        self.SE_chanl5RS = TH_Attn(128,16)

        '''S通道编码器'''
        self.stage1_S = RSU5(1,32//8,64//8)
        self.pool12_S = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2_S = RSU5(64//8,32//8,128//8)
        self.pool23_S = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3_S = RSU5(128//8,64//8,256//8)
        self.pool34_S = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4_S = RSU5(256//8,128//8,512//8)
        self.pool45_S = nn.MaxPool2d(2,stride=2,ceil_mode=True)

       
        '''R通道编码器'''
        self.stage1_R = RSU4(3,32//8,64//8)
        self.pool12_R = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.step1 = RSU4F(16,8,16)

        self.stage2_R = RSU4(16,32//8,128//8)
        self.pool23_R = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.step2 = RSU4F(32,16,32)

        self.stage3_R = RSU4(32,64//8,256//8)
        self.pool34_R = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.step3 = RSU4F(64,32,64)

        self.stage4_R = RSU4(64,128//8,512//8)
        self.pool45_R = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.step4 = RSU4F(128,64,128)
       
        '''S通道解码器'''
        self.stage6 = RSU4F(192,64,192)
        
        # decoder
        self.stage4d = RSU4(384,32,192)
        self.stage3d = RSU4(288,16,144)
        self.stage2d = RSU4(192,8,96)
        self.stage1d = RSU4(120,8,60)
        
        '''注意力模块3'''
        self.chanl4d = TH_Attn(192,32)
        self.chanl3d = TH_Attn(144,64)
        self.chanl2d = TH_Attn(96,128)
        self.chanl1d = TH_Attn(60,256)

        self.side1 = nn.Conv2d(60,1,3,padding=1)
        self.side2 = nn.Conv2d(96,1,3,padding=1)
        self.side3 = nn.Conv2d(144,1,3,padding=1)
        self.side4 = nn.Conv2d(192,1,3,padding=1)
        self.side6 = nn.Conv2d(192,1,3,padding=1)

        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore1 = nn.Upsample(scale_factor=1, mode='bilinear')

        self.outconv = nn.Conv2d(5,1,1)
        
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))

    def forward(self, R, S):

#        R = x[:,0:3,:,:]
#        S = x[:,3:4,:,:]

        hx_S = S
        #stage 1_R
        hx1_S = self.stage1_S(hx_S)
        hx_S1 = self.pool12_S(hx1_S)

        #stage 2
        hx2_S = self.stage2_S(hx_S1)
        hx_S2 = self.pool23_S(hx2_S)

        #stage 3
        hx3_S = self.stage3_S(hx_S2)
        hx_S3 = self.pool34_S(hx3_S)

        #stage 4
        hx4_S = self.stage4_S(hx_S3)
        hx_S = self.pool45_S(hx4_S)

       
        hx_R = R
        #stage 1_R
        hx1_R = self.stage1_R(hx_R)
        hx_R1 = self.pool12_R(hx1_R)

        hx_RS = self.step1(torch.cat((hx_R1,hx_S1),1))

        hx_RS_chan1 = hx_RS * self.SE_chanl1RS(hx_RS)

        #stage 2
        hx2_R = self.stage2_R(hx_RS_chan1)
        hx_R2 = self.pool23_R(hx2_R)

        hx_RS = self.step2(torch.cat((hx_R2,hx_S2),1))
        hx_RS_chan2 = hx_RS * self.SE_chanl2RS(hx_RS)

        #stage 3
        hx3_R = self.stage3_R(hx_RS_chan2)
        hx_R3 = self.pool34_R(hx3_R)

        hx_RS = self.step3(torch.cat((hx_R3,hx_S3),1))
        hx_RS_chan3 = hx_RS * self.SE_chanl3RS(hx_RS)

        #stage 4
        hx4_R = self.stage4_R(hx_RS_chan3)
        hx_R4 = self.pool45_R(hx4_R)

        hx_RS = self.step4(torch.cat((hx_R4,hx_S),1))
        hx_RS = hx_RS * self.SE_chanl4RS(hx_RS)

        #stage 6
        hx = torch.cat((hx_RS,hx_S),1)
        hx6 = self.stage6(hx)

        hx6up = self.upscore1(hx6)

        #-------------------- decoder --------------------

        hx4d = self.stage4d(torch.cat((hx6up,hx_RS,hx_S),1))
        hx4d = hx4d * self.chanl4d(hx4d)
        hx4dup = self.upscore2(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup,hx_RS_chan3,hx_S3),1))
        hx3d = hx3d * self.chanl3d(hx3d)
        hx3dup = self.upscore2(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup,hx_RS_chan2,hx_S2),1))
        hx2d = hx2d * self.chanl2d(hx2d)
        hx2dup = self.upscore2(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup,hx_RS_chan1,hx_S1),1))
        hx1d = hx1d * self.chanl1d(hx1d)
        hx1d = self.upscore2(hx1d)

        #side output
        d1 = self.side1(hx1d)
#        print(d1.size())

#        d2 = self.side2(hx2d)
#        d2 = self.upscore3(d2)
#
#        d3 = self.side3(hx3d)
#        d3 = self.upscore4(d3)
#
#        d4 = self.side4(hx4d)
#        d4 = self.upscore5(d4)
#
#        d6 = self.side6(hx6)
#        d6 = self.upscore1(d4)
#
#        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d6),1))
#        d1_0 = d1
#        
#        d1 = self.lamda*d1 + d1_0
#        d1 = self.leaky_relu(d1)

        heatmap = torch.mean(d1, axis=0)
        
        return heatmap, F.sigmoid(d1)


   
    
    