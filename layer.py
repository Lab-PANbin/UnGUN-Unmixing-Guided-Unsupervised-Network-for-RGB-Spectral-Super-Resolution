import torch
from torch import nn
import torch.nn.functional as F



class encoder_hr_hsi(nn.Module):
    def __init__(self, endmember, band_hsi):
        super(encoder_hr_hsi,self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.layer1 = nn.Linear(self.band_hsi, 2*self.band_hsi)
        self.layer2 = nn.Linear(2*self.band_hsi, self.band_hsi)
        self.layer3 = nn.Linear(self.band_hsi, int(self.band_hsi/2))
        self.layer4 = nn.Linear(int(self.band_hsi/2), self.endmember)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.out1 = self.relu(self.layer1(x))
        self.out2 = self.relu(self.layer2(self.out1))
        self.out3 = self.relu(self.layer3(self.out2))
        self.out4 = self.relu(self.layer4(self.out3))
        self.out = F.softmax(self.out4, dim=3)
        return self.out

class decoder_hsi(nn.Module):
    def __init__(self, endmember, band_hsi):
        super(decoder_hsi,self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.layer = nn.Linear(self.endmember,self.band_hsi,bias=False)

    def forward(self, x):
        self.out = self.layer(x)
        return self.out


class encoder_RGB(nn.Module):
    def __init__(self, endmember, band_RGB):
        super(encoder_RGB,self).__init__()
        self.endmember = endmember
        self.band_RGB = band_RGB
        self.layer1 = nn.Linear(self.band_RGB, 2*self.band_RGB)
        self.layer2 = nn.Linear(2*self.band_RGB, 4*self.band_RGB)
        self.layer3 = nn.Linear(4*self.band_RGB, 8*self.band_RGB)
        self.layer4 = nn.Linear(8*self.band_RGB, self.endmember)
        self.relu = nn.ReLU()
    def forward(self, x):
        self.out1 = self.relu(self.layer1(x))
        self.out2 = self.relu(self.layer2(self.out1))
        self.out3 = self.relu(self.layer3(self.out2))
        self.out4 = self.layer4(self.out3)
        self.out = F.softmax(self.out4, dim=3)
        return self.out
        

class decoder_RGB(nn.Module):
    def __init__(self, endmember, band_RGB):
        super(decoder_RGB,self).__init__()
        self.endmember = endmember
        self.band = band_RGB
        self.layer = nn.Linear(self.endmember,self.band,bias=False)
 
    def forward(self, x):
        self.out = self.layer(x)
        return self.out




class decoder_adaption(nn.Module):
    def __init__(self, band_hsi):
        super(decoder_adaption,self).__init__()
        self.band_hsi = band_hsi
        self.layer1 = nn.Conv2d(self.band_hsi, int(self.band_hsi/2), 3,1,1)
        self.layer2 = nn.Conv2d(int(self.band_hsi/2), int(self.band_hsi/2), 3,1,1)
        self.layer3 = nn.Conv2d(int(self.band_hsi/2), int(self.band_hsi/2), 3,1,1)
        self.layer4 = nn.Conv2d(int(self.band_hsi/2), self.band_hsi, 3,1,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,3,1,2)
        self.out = self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))
        x = x.permute(0,2,3,1)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)
class reshape1(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.size(0), -1,1,1)
class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,kernel_size=3,stride=1,act=True,dropout = False):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=RELU,BN=BN,spatial_norm=True,stride=1)
        self.act = act
        self.actlayer = BCR(kernel=kernel_size,cin=cout,cout=cout,group=cout,RELU=RELU,padding=(kernel_size-1)//2,BN=BN,spatial_norm=True,stride=stride)
        if dropout == True:
            self.dropout = nn.Dropout2d(0.1)
        self.drop = dropout
    def forward(self, x):
        if self.drop:
            [B,C,H,W] = x.shape
            x = x.permute([0,2,3,1]).reshape([B*H*W,C,1,1])
            x = self.dropout(x)
            x = x.reshape([B,H,W,C]).permute([0,3,1,2])
        output = self.compressLayer(x)
        if self.act == True:
            output = self.actlayer(output)

        return output
class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False,spatial_norm = False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.Swish
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Dis_stage(nn.Module):
    def __init__(self,cin=31,cout=64,down=True):
        super(Dis_stage, self).__init__()
        self.down = down
        if down:
            self.downsample = nn.Sequential(
                denselayer(cin=cin,cout=cout,RELU=True,kernel_size=3,stride=2,BN=True),
                denselayer(cin=cout,cout=cout,RELU=True,kernel_size=3,stride=2,BN=True),
            )
        else:
            self.downsample = nn.Sequential(
                denselayer(cin=cin,cout=cout,RELU=True,kernel_size=3,stride=1,BN=True))
        self.denseconv = nn.Sequential(
            denselayer(cin = cout, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*2, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*3, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*4, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*5, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*6, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*7, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
        )

    def forward(self,MSI):
        if self.down:
            dfeature = self.downsample(MSI)
        else:
            dfeature = self.downsample(MSI)

        feature = [dfeature]

        for conv in self.denseconv:
            feature.append(conv(torch.cat(feature,dim=1)))

        return feature[-1] + dfeature

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Discriminator_HSI(nn.Module):
    def __init__(self, band_hsi):
        super(Discriminator_HSI, self).__init__()
        self.band_hsi = band_hsi
        self.stage1 = Dis_stage(cin=self.band_hsi,cout=64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            reshape1(),
            denselayer(cin = 256,cout=1,RELU=False,BN=False,kernel_size=1,stride=1),
            Flatten(),
            nn.Sigmoid())
    def forward(self,HSI):
        feature = self.stage1(HSI)
        prod = self.classifier(feature)
    
        return prod


def kl_divergence(p, q):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    
    return s1 + s2

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


def conv(in_channels, out_channels, kernel_size, bias=True, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)





class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=50):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:,1:,:,:])
        c_tv = torch.pow((x[:,1:,:,:]-x[:,:c_x-1,:,:]),2).sum()
        return self.TVLoss_weight*2*(c_tv/count_c)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]




















