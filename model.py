import layer
from torch import nn
import torch
from torch.autograd import Function
import torch.autograd as ag
import math

class Network(nn.Module):
    def __init__(self, endmember, band_hsi, band_RGB):
        super(Network,self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.band_RGB = band_RGB
        self.gradnorm = GradNorm().apply
        self.gene = Generator(self.endmember, self.band_hsi, self.band_RGB)
        self.dis = Discriminator(self.band_hsi)

    def forward(self, Input_1, Input_2):
        
        self.abundance_hsi, self.abundance_RGB, self.hrhsi, self.RGB, self.pred_hrhsi = self.gene(Input_1, Input_2)

        Input_1 = Input_1.permute(0,3,1,2)
        Input_2 = Input_2.permute(0,3,1,2)
        
        self.dis_hsi2, self.dis_pre2 = self.dis(self.gradnorm(Input_1, torch.ones(1, device = Input_1.device)), self.gradnorm(self.pred_hrhsi, torch.ones(1, device = self.pred_hrhsi.device)))
        
        return self.abundance_hsi, self.abundance_RGB, self.hrhsi, self.RGB, self.pred_hrhsi, self.dis_hsi2, self.dis_pre2



class GradNorm(Function):
    @staticmethod
    def forward(ctx, input_x,scale):
        ctx.save_for_backward(scale)
        return input_x
    @staticmethod
    def backward(ctx, grad_output):
        [B,C,H,W] = grad_output.shape
        scale, = ctx.saved_tensors
        gradnrom = (grad_output**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        stdnrom = 1/math.sqrt(C*H*W)
        gradnrom = stdnrom / gradnrom

        gradnrom = torch.clamp(gradnrom,min=0,max=1)

        grad_output = gradnrom * grad_output
        grad_output = scale * grad_output

        return grad_output,None


class Generator(nn.Module):
    def __init__(self, endmember, band_hsi, band_RGB):
        super(Generator,self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.band_RGB = band_RGB

        self.encoder_hr_hsi = layer.encoder_hr_hsi(self.endmember, self.band_hsi)
        self.decoder_hsi = layer.decoder_hsi(self.endmember, self.band_hsi)
        self.encoder_RGB = layer.encoder_RGB(self.endmember, self.band_RGB)
        self.decoder_RGB = layer.decoder_RGB(self.endmember, self.band_RGB)
        self.decoder_adaption = layer.decoder_adaption(self.band_hsi)


    def forward(self, hsi, RGB):

        self.abundance_hsi = self.encoder_hr_hsi(hsi)
        self.hrhsi = self.decoder_hsi(self.abundance_hsi)
        self.abundance_RGB = self.encoder_RGB(RGB)
        self.RGB = self.decoder_RGB(self.abundance_RGB)
        self.pred_hrhsi_hat = self.decoder_hsi(self.abundance_RGB)
        self.pred_hrhsi = self.decoder_adaption(self.pred_hrhsi_hat)
        self.hrhsi1 = self.hrhsi.permute(0,3,1,2)
        self.RGB1 = self.RGB.permute(0,3,1,2)
        self.pred_hrhsi1 = self.pred_hrhsi.permute(0,3,1,2)



        return self.abundance_hsi, self.abundance_RGB, self.hrhsi1, self.RGB1, self.pred_hrhsi1


class Discriminator(nn.Module):
    def __init__(self, band_hsi):
        self.band_hsi = band_hsi
        super(Discriminator,self).__init__()
        self.discriminator = layer.Discriminator_HSI(self.band_hsi)

    def forward(self, hsi, pred_hrhsi):
    
        self.dis_hsi = self.discriminator(hsi)
        self.dis_pre = self.discriminator(pred_hrhsi)

        return self.dis_hsi, self.dis_pre


def calc_gradient_penalty(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=.5, device=None):

        
        if alpha is not None:
            alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
        else:
            alpha = torch.rand(real_data.size(0), 1, device=device)
        alpha = torch.reshape(alpha,[real_data.size(0), 1, 1, 1])
        alpha = alpha.expand(real_data.size())
        fake_data = torch.nn.functional.interpolate(fake_data, size=[512,217], scale_factor=None, mode='nearest', align_corners=None)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)
        

        gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        [B,C,H,W] = gradients.shape
        
        gradients = torch.reshape(gradients,[B,C*H*W])
        gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA

        return gradient_penalty