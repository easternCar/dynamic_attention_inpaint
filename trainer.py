import os
import torch
import torch.nn as nn
from torch import autograd
from model.network import Generator, SN_Patch_Discriminator
from model.perceptual import VGG16FeatureExtractor
from torch.autograd import Variable
import torch.nn.functional as F
from utils.tools import get_model_list
from utils.logger import get_logger
from torch.nn import Parameter


torch.autograd.set_detect_anomaly(True)

logger = get_logger()

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.dataset_name = self.config['dataset_name']

        # ---- Generator ----------------------------
        self.G = Generator(self.config['netG'])
        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        self.P = VGG16FeatureExtractor()

        # ---- Discriminator --------------------------
        self.globalD = SN_Patch_Discriminator(self.config['netD'])

        d_params = list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        if self.use_cuda:
            self.G.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])
            self.P.to(self.device_ids[0])



    def forward(self, x, masks, ground_truth):
        # x : incomplete image
        # masks : binary mask
        # ground_truth : original image

        self.train()
        losses = {}

        # <g line>
        x2, x1, atts = self.G(x, masks)

        # PATCH GENERATED
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        # <d line>
        global_real_pred, global_fake_pred = \
                self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())

        # PATCHGAN
        losses['wgan_d'] = torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = global_penalty
        losses['l1'] = nn.L1Loss()(x1 * (1. - masks), ground_truth * (1. - masks)) * self.config['coarse_l1_alpha'] \
                       + nn.L1Loss()(x2 * (1. - masks), ground_truth * (1. - masks))

        # Perceptual loss
        # Get the deep semantic feature maps, and compute Perceptual Loss
        losses['percep'] = self.calc_percep_loss(x2_inpaint, ground_truth)

        # DAM loss
        losses['att'] = self.calc_attention_loss(ground_truth, masks, x2_inpaint, atts)

        # <d line 2>
        global_real_pred, global_fake_pred = \
            self.dis_forward(self.globalD, ground_truth, x2_inpaint)
        losses['wgan_g'] = -torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        
        return losses, x1_inpaint, x2_inpaint


    # DAM loss func
    def calc_attention_loss(self, gt, mask, out, attmaps):
        levels = len(attmaps) # originally,  = 4
        att_loss = 0.0

        for layeridx in range(levels):
            # gt difference 
            layer_attmap = attmaps[layeridx]
            layer_inpaint = F.interpolate(out, size=(layer_attmap.size(2), layer_attmap.size(3)), mode='nearest')

            # gt resize
            layer_gt = F.interpolate(gt, size=(layer_inpaint.size(2), layer_inpaint.size(3)), mode='nearest')
            gt_diff = torch.mean(torch.abs(layer_gt - layer_inpaint), dim=1, keepdim=True)

            #print(str(layeridx) + "_gt_diff " + str(gt_diff.size()))
            #print(str(layeridx) + "_attmap " + str(layer_attmap.size()))
            
            layer_mask = F.interpolate(mask, size=(layer_inpaint.size(2), layer_inpaint.size(3)), mode='nearest')
            #print(str(layeridx) + "_layermask " + str(layer_mask.size()))
            att_loss = att_loss + nn.L1Loss()(gt_diff * (1. - layer_mask), layer_attmap * (1. - layer_mask))

        return att_loss


    # Perceptual loss func
    def calc_percep_loss(self, output, gt):
        feat_output = self.P(output)
        feat_gt = self.P(gt)

        loss_p = 0.0
        for i in range(3):
            loss_p += nn.L1Loss()(feat_output[i], feat_gt[i])

        return loss_p

        

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        #real_pred = netD(ground_truth)
        #fake_pred = netD(x_inpaint)


        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size, channel, height, width = real_data.size()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() // batch_size)).contiguous() \
            .view(batch_size, channel, height, width)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)


        grad_outputs = torch.ones(disc_interpolates.size())
        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]


        gradients = gradients.contiguous().view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def calculate_gradient_penalty_v2(self, netD, real_data, fake_data):
        batch_size, channel, height, width = real_data.size()
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, channel, height, width)
        if self.use_cuda:
            eta = eta.cuda()

        interpolated = eta * real_data + ((1 - eta) * fake_data)

        if self.use_cuda:
            interpolated = interpolated.cuda()
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = netD(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty


    def inference(self, x, masks):
        '''
        self.eval()
        #x1, x2, offset_flow = self.netG(x, masks)
        x1, x2, offset_flow, infer_arr, pyramid_layers = self.netG(x, masks)

        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow
        '''
        return None, None

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.G.state_dict(), gen_name)
        #torch.save({'localD': self.localD.state_dict(), 'globalD': self.globalD.state_dict()}, dis_name)
        torch.save(self.globalD.state_dict(), dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(), 'dis': self.optimizer_d.state_dict()}, opt_name)

      

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        #self.netG.load_state_dict(torch.load(last_model_name), strict=False)    # 1
        self.load_my_state_dict(self.G, torch.load(last_model_name))       # 2

        iteration = int(last_model_name[-11:-3])


        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            #self.localD.load_state_dict(state_dict['localD'])
            #self.globalD.load_state_dict(state_dict['globalD'])
            self.globalD.load_state_dict(state_dict)
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            #self.optimizer_g.load_state_dict(state_dict['gen'])    # 1
            #self.load_my_state_dict(self.optimizer_g, state_dict['gen']) # 2

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration


    def load_my_state_dict(self, our_model, state_dict):
        own_state = our_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)
