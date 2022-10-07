import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
import cv2
from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding, LayerResult


from model.conv_func import gen_conv, ResnetBlock, Conv2dBlock, DisConvModule



class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']

        self.coarse_generator = CoarseGenerator(self.input_dim,  self.cnum)
        self.daw_generator = DAW_Generator(self.input_dim, self.cnum)

    def forward(self, x, mask):
        # x : input
        # mask : binary mask

        # gt : gt (only when using attention map)
        # fr_network : ResNet (only when using attention map)


        # 1) Occlusion Generation
        x1_out = self.coarse_generator(x, mask)
        x2_out, daw_maps = self.daw_generator(x1_out, mask)

        return x2_out, x1_out, daw_maps




class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(CoarseGenerator, self).__init__()
        OUT_CHANNEL = 3

        # encoder
        # 128 * 128 * cnum
        self.conv1_1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv1_2 = gen_conv(cnum, cnum, 3, 1, 1)
        self.conv1_down = gen_conv(cnum, cnum * 2, 3, 2, 1)

        # 64 * 64 * cnum2
        self.conv2_1 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv2_2 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv2_down = gen_conv(cnum*2, cnum * 4, 3, 2, 1)

        # 32 * 32 * cnum4
        self.conv3_1 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv3_2 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv3_down = gen_conv(cnum*4, cnum * 8, 3, 2, 1)

        # 16 * 16 * cnum8
        self.conv4_1 = gen_conv(cnum*8, cnum*8, 3, 1, 1)
        self.conv4_2 = gen_conv(cnum*8, cnum*8, 3, 1, 1)

        # decoder
        # 16 * 16 * cnum8
        self.conv5_1 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.conv5_2 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        # 32 * 32 * cnum4
        self.conv6_1 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv6_2 = gen_conv(cnum*2, cnum*2, 3, 1, 1)

        # 64 * 64 * cnum2
        self.conv7_1 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.conv7_2 = gen_conv(cnum, cnum, 3, 1, 1)

        # 128 * 128 * cnum
        self.conv8_1 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.conv8_2 = gen_conv(cnum // 2, cnum // 2, 3, 1, 1)
        self.conv8_out = gen_conv(cnum // 2, OUT_CHANNEL, 3, 1, 1, activation='none')

        # ETC
        self.leakyRelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()




    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        ones = ones.cuda()

        # 5 x 128 x 128
        x = self.conv1_1(torch.cat([x, ones, mask], dim=1))
        x = self.conv1_2(x)
        x = self.conv1_down(x)

        # cnum*2 x 64 x 64
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_down(x)

        # cnum*4 x 32 x 32
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_down(x)

        # cnum*8 x 16 x 16
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # cnum*4 x 32 x 32
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # cnum*4 x 64 x 64
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # cnum*4 x128 x 128
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.conv8_out(x)
        x_out = torch.clamp(x, -1., 1.)

        # 16x16, 32x32, 64x64, 128x128
        return x_out



# for completion
class DAW_Generator(nn.Module):
    def __init__(self, input_dim, cnum, use_unet=True):
        super(DAW_Generator, self).__init__()
        self.use_unet = use_unet
        OUT_CHANNEL = 3
        
        if use_unet:
            skip_factor = 2
        else:
            skip_factor = 1

        # input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1

        # encoder
        # 128 * 128 * cnum
        self.conv1_1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv1_2 = ResnetBlock(cnum, 3, 1, 1)
        self.conv1_down = gen_conv(cnum, cnum * 2, 3, 2, 1)

        # 64 * 64 * cnum2
        self.conv2_1 = ResnetBlock(cnum*2, 3, 1, 1)
        self.conv2_down = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)

        # 32 * 32 * cnum4
        self.conv3_1 = ResnetBlock(cnum*4, 3, 1, 1)
        self.conv3_down = gen_conv(cnum*4, cnum * 8, 3, 2, 1)

        # 16 * 16 * cnum8
        self.conv4_1 = ResnetBlock(cnum*8, 3, 1, 1)
        self.conv4_atrous1 = gen_conv(cnum*8, cnum*8, 3, 1, 2, rate=2)
        self.conv4_atrous2 = gen_conv(cnum*8, cnum*8, 3, 1, 4, rate=4)
        self.conv4_atrous3 = gen_conv(cnum*8, cnum*8, 3, 1, 8, rate=8)
        self.conv4_atrous4 = gen_conv(cnum*8, cnum*8, 3, 1, 16, rate=16)
        self.conv4_2 = ResnetBlock(cnum*8, 3, 1, 1)
        

        # 16 * 16 * cnum*8 -> 32 * 32 * cnum * 4
        self.conv5_1 = ResnetBlock(cnum*8, 3, 1, 1)
        self.conv5_DAW = DAW_Block(cnum*8, cnum*4, layer_idx=0)
        #<upsample>

        # 32 * 32 * cnum*4 -> 64 * 64 * cnum * 2
        self.conv6_1 = ResnetBlock(cnum*4, 3, 1, 1)
        self.conv6_DAW = DAW_Block(cnum*4, cnum*2, layer_idx=1)
        #<upsample>

        # 64 * 64 * cnum*8 -> 128 * 128 * cnum
        self.conv7_1 = ResnetBlock(cnum*2, 3, 1, 1)
        self.conv7_DAW = DAW_Block(cnum*2, cnum, layer_idx=2)
        #<upsample>

        # 64 * 64 * cnum*4 -> 128 * 128 * cnum * 2
        self.conv8_1 = ResnetBlock(cnum, 3, 1, 1)
        self.conv8_DAW = DAW_Block(cnum, 3, layer_idx=3)

        # ETC
        self.leakyRelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()





    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        ones = ones.cuda()

        x = torch.cat([x, ones, mask], dim=1)

        # encoder (128x128 ---> 64x64)
        x = self.conv1_1(x)
        x = self.conv1_2(x)  # ---- skip 1 (128x128)
        skip1_x = x.clone()
        x = self.conv1_down(x)

        # (64x64 ---> 32x32)
        x = self.conv2_1(x)  # ---- skip 2 (64x64)
        skip2_x = x.clone()
        x = self.conv2_down(x)
        
        # (32x32 ---> 16x16)
        x = self.conv3_1(x)  # ---- skip 3 (32x32)
        skip3_x = x.clone()
        x = self.conv3_down(x)

        # (16x16 + dilated)
        x = self.conv4_1(x)  # ---- skip 4 (16x16)
        skip4_x = x.clone()
        x = self.conv4_atrous1(x)
        x = self.conv4_atrous2(x)
        x = self.conv4_atrous3(x)
        x = self.conv4_atrous4(x)
        x = self.conv4_2(x)

        # (16x16 ---> 32x32)
        x = self.conv5_1(x)
        x, mask_16x16 = self.conv5_DAW(x=x, prev_x=skip4_x, prev_mask=None)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # (32x32 ---> 64x64)
        x = self.conv6_1(x)
        x, mask_32x32 = self.conv6_DAW(x=x, prev_x=skip3_x, prev_mask=mask_16x16)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # (64x64 ---> 128x128)
        x = self.conv7_1(x)
        x, mask_64x64 = self.conv7_DAW(x=x, prev_x=skip2_x, prev_mask=mask_32x32)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # (128x128 ---> OUT)
        x = self.conv8_1(x)
        x, mask_128x128 = self.conv8_DAW(x=x, prev_x=skip1_x, prev_mask=mask_64x64)
        x = torch.clamp(x, -1., 1.)

        return x, [mask_16x16, mask_32x32, mask_64x64, mask_128x128]


class DAW_Block(nn.Module):
    def __init__(self, input_dim, output_dim, layer_idx):
        super(DAW_Block, self).__init__()
        self.layer_idx = layer_idx

        # if layer_idx = 0 (16) ---> no previous mask
        # if layer_idx = 3 (128) ---> no need to send mask

        # 3x3 conv + batchnorm + relu (because of skip-connection, dim*2 ---> dim)
        self.in_conv1 = gen_conv(input_dim*2, input_dim, 3, 1, 1)

        # 3x3 conv + batchnorm + relu
        self.mask_conv1 = gen_conv(input_dim, input_dim // 2, 3, 1, 1)
        self.mask_conv2 = gen_conv(input_dim // 2, 1, 3, 1, 1, activation='none')

        # (if 1st layer, input_dim --> input_dim / 2 & no upsample)
        # (if not, input_dim * 2 --> input_dim / 2 & with upsample)
        if layer_idx == 0:
            out_factor = 1
        else:
            out_factor = 2

        self.out_conv1 = gen_conv(input_dim, output_dim, 3, 1, 1)

        # ETC
        self.leakyRelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        

    def forward(self, x, prev_x, prev_mask):

        # prev_mask : from previous daw, mask
        # prev_x : from skip-connection
        if prev_mask is None:
            assert self.layer_idx == 0
        
        # (1) CONCAT SKIP CONNECTION -=============
        
        #x_orig = x.clone()
        x_fea = self.in_conv1(torch.cat([x, prev_x], dim=1))       # concatenated f -> f
        
        # f to mask
        x_mask = self.mask_conv1(x_fea)
        x_mask = torch.clamp(x_mask, -1., 1.)
        x_mask = self.sigmoid(self.mask_conv2(x_mask)).clone()

        # (2) MASKING -=============
        # x * mask
        x_fea_weighted = self.leakyRelu(x_fea) * x_mask

        # x + (x * mask)
        x_out = x_fea_weighted + x_fea

        
        x_out = self.out_conv1(x_out)
        x_out = torch.clamp(x_out, -1., 1.)
        
        return x_out, x_mask
        


# for completion
class Hybrid_Generator(nn.Module):
    def __init__(self, input_dim, cnum, use_unet=True, use_cuda=True, device_ids=None):
        super(Hybrid_Generator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.use_unet = use_unet
        OUT_CHANNEL = 3
        
        if use_unet:
            skip_factor = 2
        else:
            skip_factor = 1

        # input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1

        # encoder
        # 128 * 128 * 3 
        self.resblk1 = ResnetBlock(input_dim, 3, 1, 1)

        # 128 * 128 * 3 -> 64 * 64 * cnum
        self.downblk1 = DownsampleBlock(input_dim, cnum)
        self.resblk2 = ResnetBlock(cnum, 3, 1, 1)

        # 64 * 64 * cnum -> 32 * 32 * cnum * 2
        self.downblk2 = DownsampleBlock(cnum, cnum*2)
        self.resblk3 = ResnetBlock(cnum*2, 3, 1, 1)

        # 32 * 32 * cnum * 2 -> 16 * 16 * cnum * 4
        self.downblk3 = DownsampleBlock(cnum*2, cnum*4)
        self.resblk4 = ResnetBlock(cnum*4, 3, 1, 1)


        # down conv (-> 1/2)
        self.downconv = gen_conv(cnum * 4, cnum * 8, 3, 2, 1)

        

        # 16 * 16 * cnum * 4 -> 8 * 8 * cnum*8 (4 HDC)
        self.hdc1 = HDCBlock(cnum*8, cnum*8)
        self.hdc2 = HDCBlock(cnum*8, cnum*8)
        self.hdc3 = HDCBlock(cnum*8, cnum*8)
        self.hdc4 = HDCBlock(cnum*8, cnum*8)

        # 8 * 8 * cnum*8 -> 16 * 16 * cnum * 4
        self.upblk1 = UpsampleBlock(cnum*8, cnum*4)
        self.resblk5 = ResnetBlock(cnum*4*skip_factor, 3, 1, 1)

        # 16 * 16 * cnum*4 -> 32 * 32 * cnum * 2
        self.upblk2 = UpsampleBlock(cnum*4*skip_factor, cnum*2)
        self.resblk6 = ResnetBlock(cnum*2*skip_factor, 3, 1, 1)

        # 32 * 32 * cnum*8 -> 64 * 64 * cnum
        self.upblk3 = UpsampleBlock(cnum*2*skip_factor, cnum)
        self.resblk7 = ResnetBlock(cnum*skip_factor, 3, 1, 1)

        # 64 * 64 * cnum*4 -> 128 * 128 * cnum * 2
        self.upblk4 = UpsampleBlock(cnum*skip_factor, 3)
        #self.resblk8 = ResnetBlock(cnum // 2, 3, 1, 1)



        # ETC
        self.leakyRelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()





    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()

        x = torch.cat([x, ones, mask], dim=1)

        # encoder
        x = self.resblk1(x)
        x = self.downblk1(x)    
        x = self.resblk2(x)     # ---- skip 1 (64x64)
        if self.use_unet:
            skip1_x = x.clone()
        x = self.downblk2(x)
        x = self.resblk3(x)     # ---- skip 2 (32x32)
        if self.use_unet:
            skip2_x = x.clone()
        x = self.downblk3(x)

        x = self.resblk4(x)     # ---- skip 3 (16x16)
        if self.use_unet:
            skip3_x = x.clone()
        
        # down con
        x = self.downconv(x)

        # center
        x = self.hdc1(x)
        x = self.hdc2(x)
        x = self.hdc3(x)
        x = self.hdc4(x)
        
        # up conv


        # decoder
        x = self.upblk1(x)
        if self.use_unet:
            #print("1 --- " + str(x.size()))
            #print(skip3_x.size())
            x = torch.cat([x, skip3_x], dim=1)

        x = self.resblk5(x)     # ---- skip 3

        x = self.upblk2(x)
        if self.use_unet:
            x = torch.cat([x, skip2_x], dim=1)

        x = self.resblk6(x)     # ---- skip 2
        x = self.upblk3(x)
        if self.use_unet:
            x = torch.cat([x, skip1_x], dim=1)

        x = self.resblk7(x)     # ---- skip 1
        x = self.upblk4(x)
        #x = self.resblk8(x)

        return x


class LaplacianBlock(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(LaplacianBlock, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        OUT_CHANNEL = 3

        # 3x3 conv + batchnorm + relu
        self.conv1 = gen_conv(input_dim, input_dim // 2, 3, 1, 1)
        self.conv2 = gen_conv(input_dim // 2, cnum, 3, 1, 1, activation='none')
        
    def forward(self, x):

        #x_orig = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class DownsampleBlock(nn.Module):
    def __init__(self, input_dim, cnum):
        super(DownsampleBlock, self).__init__()
        OUT_CHANNEL = 3

        # 3x3 conv + batchnorm + relu
        self.conv1 = gen_conv(input_dim, cnum, 3, 1, 1)
        # avg 2d pool
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        # 3x3 conv + batchnorm + no relu
        self.conv2 = gen_conv(cnum, cnum, 3, 1, 1, activation='none')
        
    def forward(self, x):

        #x_orig = x.clone()
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)

        return x



class UpsampleBlock(nn.Module):
    def __init__(self, input_dim, cnum):
        super(UpsampleBlock, self).__init__()
        OUT_CHANNEL = 3

        # 3x3 conv + batchnorm + relu
        self.conv1 = gen_conv(input_dim, cnum, 3, 1, 1)
        # upsampling
        # 3x3 conv + batchnorm + no relu
        self.conv2 = gen_conv(cnum, cnum, 3, 1, 1, activation='none')
        
    def forward(self, x):

        #x_orig = x.clone()
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')    # upsample
        x = self.conv2(x)

        return x


class HDCBlock(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(HDCBlock, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        OUT_CHANNEL = 3

        #self.conv0 = gen_conv(input_dim, cnum, 1, 1, 0)

        # 3x3 conv + batchnorm + relu
        self.conv1 = gen_conv(input_dim, cnum, 3, 1, 1, rate=1)
        
        # 3x3 conv + batchnorm + relu + dilation 2
        self.conv2 = gen_conv(cnum, cnum, 3, 1, 2, rate=2)
        
        # 3x3 conv + batchnorm + relu + dilation 3
        self.conv3 = gen_conv(cnum, cnum, 3, 1, 3, rate=3)


        
    def forward(self, x):

        #print(x.size())
        #x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x







class SN_Patch_Discriminator(nn.Module):
    def __init__(self, config):
        super(SN_Patch_Discriminator, self).__init__()

        self.channels = config['input_dim']
        self.leak = 0.1
        self.w_g = 16

        self.ndf = config['ndf']

        self.conv1 = Conv2dBlock(self.channels, self.ndf, 3, 1, 1, activation='lrelu', weight_norm='sn')
        self.conv2 = Conv2dBlock(self.ndf, self.ndf, 4, 2, 1, activation='lrelu', weight_norm='sn')
        self.conv3 = Conv2dBlock(self.ndf, self.ndf*2, 3, 1, 1, activation='lrelu', weight_norm='sn')
        self.conv4 = Conv2dBlock(self.ndf*2, self.ndf*2, 4, 2, 1, activation='lrelu', weight_norm='sn')
        self.conv5 = Conv2dBlock(self.ndf*2, self.ndf*4, 3, 1, 1, activation='lrelu', weight_norm='sn')
        self.conv6 = Conv2dBlock(self.ndf*4, self.ndf*4, 4, 2, 1, activation='lrelu', weight_norm='sn')
        self.conv7 = Conv2dBlock(self.ndf*4, self.ndf*8, 3, 1, 1, activation='lrelu', weight_norm='sn')

        self.linear = spectral_norm_fn(nn.Linear(self.w_g * self.w_g * 512, 1))

    def forward(self, x):
        m = x
        m = self.conv1(m)
        m = self.conv2(m)
        m = self.conv3(m)
        m = self.conv4(m)
        m = self.conv5(m)
        m = self.conv6(m)
        m = self.conv7(m)

        return self.linear(m.view(m.size()[0], -1))

class PatchGAN_Dis(nn.Module):
    def __init__(self, use_cuda=True, device_ids=None):
        super(PatchGAN_Dis, self).__init__()
        self.input_dim = 3
        self.cnum = 64
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x
