import argparse
import yaml
from dataloader import forming_dataloader, dataloader_official_version
from torch.utils import data
from models.models import *
from models.vit_model import vit_discriminator
from models.normal_vit import VisionTransformer
from utils.VTGAN_loss import *
import random
import torch.optim as optim
from torch import nn
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from visdom import Visdom
from utils.visualization import summarize_performance, iter_summarize_performance

viz = Visdom()


def get_y2(is_nor, batchsize):
    if is_nor[0]:
        y2 = torch.tensor([0.9,0.])
        y2_cache = torch.tensor([0.9,0.])
    else:
        y2 = torch.tensor([0.,0.9])
        y2_cache = torch.tensor([0.,0.9])
                
    for _ in range(batchsize-1):
        y2 = torch.vstack((y2, y2_cache))
    if batchsize == 1:
        y2 = y2[None]
    y2 = convert_to_cuda(y2)
    return y2

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config
    

        
def main(args):
    train_config = load_config(args.model_config_path)
    BATCHSIZE = train_config['batchsize']
    EPOCHS = train_config['epoch']
    nlr=0.0002
    nbeta1=0.5
    viz = Visdom()
   
    mse = nn.MSELoss()
    l1_loss = nn.L1Loss()
    ca_loss = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    hinge_loss = MyHingeLoss()
    
    if train_config["is_official"]:
        data_path = train_config["official_data_path"]
        F_A_dataset = dataloader_official_version.Official_Fundus_angio_dataset(data_path)
        F_A_dataloader = data.DataLoader(F_A_dataset, BATCHSIZE, shuffle=train_config['to_shuffle'])
        val_dataloader = data.DataLoader(F_A_dataset, 1, shuffle=train_config['to_shuffle'])
        val_iter = iter(val_dataloader)
    else:
        data_path = train_config['data_path']
        F_A_dataset = forming_dataloader.Fundus_angio_dataset(data_path)
        F_A_dataloader = data.DataLoader(F_A_dataset, BATCHSIZE, shuffle=train_config['to_shuffle'])
        
    
    # since the official code doesn't use normal vision transformer, I provide a choice.
    if train_config['normal_vit']:
        d_model1 = VisionTransformer(512, 64, 4, 2, 2048, 4, 4)
        d_model1 = nn.DataParallel(d_model1)
        d_model1 = d_model1.cuda()
        d_model2 = VisionTransformer(256, 32, 4, 2, 512, 4, 4)
        d_model2 = nn.DataParallel(d_model2)
        d_model2 = d_model2.cuda()

    else:
        d_model1 = nn.DataParallel(vit_discriminator(64)).cuda()
        d_model2 = nn.DataParallel(vit_discriminator(32)).cuda()
        
        
    # official code use separable conv2d to avoid high memory requiring, 
    # however, it doesn't take a lot if we use conv2d instead 
    if train_config['use_separable'] == False:
        g_model_coarse = nn.DataParallel(coarse_generator()).cuda()
        g_model_fine = nn.DataParallel(fine_generator()).cuda()
    else:
        g_model_coarse = nn.DataParallel(coarse_generator(use_separable=True)).cuda()
        g_model_fine = nn.DataParallel(fine_generator(use_separable=True)).cuda()
        
    

    len_along_epoch = len(F_A_dataloader)
    
    optimizerD_f = optim.Adam(d_model1.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD_c = optim.Adam(d_model2.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerG_f = optim.Adam(g_model_fine.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerG_c = optim.Adam(g_model_coarse.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    d_lr_decay_f = torch.optim.lr_scheduler.ExponentialLR(optimizerD_f, 0.99)
    d_lr_decay_c = torch.optim.lr_scheduler.ExponentialLR(optimizerD_c, 0.99)

    count = 0
    viz.line([[5.],[5.],[70.]], [0], win="VTGAN_LOSS", opts=dict(title='loss',
                                            legend=['d_f_loss', 'd_c_loss', 'gan_loss']))

    
    
    
    
    for epoch in range(EPOCHS):
        D_f_loss = 0
        D_c_loss = 0
        Gan_loss = 0
        
        for i in range(len_along_epoch):
            
            
            if train_config["is_official"]:
                iter_F_A = iter(F_A_dataloader)
                
            else:
                datachoice = [0, 1]
                ret = random.choice(datachoice)
                X_realA, X_realB, X_realA_half, X_realB_half, is_nor = next(iter_F_A)[ret]
            
            
            
            
            #need to remain that there exist batch dimension so that we take the first element of "is_nor"
            
            
            
#train the FINE descriminator------------------------------------------------------------ 
            for _ in range(2):
                X_realA, X_realB, X_realA_half, X_realB_half, is_nor = next(iter_F_A)
                
                X_realA = convert_to_cuda(X_realA)
                X_realB = convert_to_cuda(X_realB)
                X_realA_half = convert_to_cuda(X_realA_half)
                X_realB_half = convert_to_cuda(X_realB_half)
                
                y2 = get_y2(is_nor, BATCHSIZE)
                           
                optimizerD_f.zero_grad()
                d_feat1_real = d_model1(X_realA, X_realB)
                y1 = convert_to_cuda(-1*torch.ones_like(d_feat1_real[0]))
                # d_loss1 = mse(d_feat1_real[0], y1) + ca_loss(d_feat1_real[1], y2)
                d_loss1 = mse(d_feat1_real[0], y1)
                
                optimizerD_f.zero_grad()
                X_fakeB_half, x_global = g_model_coarse(X_realA_half)
                X_fakeB = g_model_fine(X_realA, x_global)
                d_feat1_fake = d_model1(X_realA, X_fakeB.detach())
                y1_fine = convert_to_cuda(torch.ones_like(d_feat1_fake[0]))
                # d_loss2 = mse(d_feat1_fake[0], y1_fine) + ca_loss(d_feat1_fake[1], y2)
                d_loss2 = mse(d_feat1_fake[0], y1_fine)
                
                d_f_loss = d_loss1 + d_loss2
                d_f_loss.backward()
                optimizerD_f.step()

                

#train the COARSE descriminator-----------------------------------------------------------------            
                optimizerD_c.zero_grad()
                d_feat2_real = d_model2(X_realA_half, X_realB_half)
                y1 = convert_to_cuda(-1*torch.ones_like(d_feat2_real[0]))
                # d_loss3 = mse(d_feat2_real[0], y1) + ca_loss(d_feat2_real[1], y2)
                d_loss3 = mse(d_feat2_real[0], y1)
                
                d_feat2_fake = d_model2(X_realA_half, X_fakeB_half.detach())
                y1_coarse = convert_to_cuda(torch.ones_like(d_feat2_fake[0]))
                # d_loss4 = mse(d_feat2_fake[0], y1_coarse) + ca_loss(d_feat2_fake[1], y2)
                d_loss4 = mse(d_feat2_fake[0], y1_coarse)
                
                
                d_c_loss = d_loss3+d_loss4
                d_c_loss.backward()
                optimizerD_c.step()
                
            
                
            X_realA, X_realB, X_realA_half, X_realB_half, is_nor = next(iter_F_A)
            X_realA = convert_to_cuda(X_realA)
            X_realB = convert_to_cuda(X_realB)
            X_realA_half = convert_to_cuda(X_realA_half)
            X_realB_half = convert_to_cuda(X_realB_half)
            y2 = get_y2(is_nor, BATCHSIZE)
            
            optimizerG_f.zero_grad()
            optimizerG_c.zero_grad()
            X_fakeB_half, x_global = g_model_coarse(X_realA_half)
            X_fakeB = g_model_fine(X_realA, x_global)
            g_f_loss = mse(X_fakeB, X_realB)
            
            
            g_c_loss = mse(X_fakeB_half, X_realB_half)
            g_total_loss = g_f_loss+g_c_loss
            
            g_total_loss.backward()
            optimizerG_f.step()
            optimizerG_c.step()
            
#train the FINE and COARSE together as a gan model-------------------------------------------------------------
            
            X_fakeB_half, x_global = g_model_coarse(X_realA_half)
            X_fakeB = g_model_fine(X_realA, x_global.detach())
            
            d_feat1_fake = d_model1(X_realA, X_fakeB)
            d_feat1_real = d_model1(X_realA, X_realB)
            
            d_feat2_real = d_model2(X_realA_half, X_realB_half)
            d_feat2_fake = d_model2(X_realA_half, X_fakeB_half)
            
            y1 = convert_to_cuda(-1*torch.ones_like(d_feat1_fake[0]))
            y1_half = convert_to_cuda(-1*torch.ones_like(d_feat2_fake[0]))
            
            X_realB_stacked = torch.cat([X_realB, X_realB, X_realB], dim=1)
            X_fakeB_stacked = torch.cat([X_fakeB, X_fakeB, X_fakeB],dim=1)
            X_realB_half_stacked = torch.cat([X_realB_half, X_realB_half, X_realB_half], dim=1)
            X_fakeB_half_stacked = torch.cat([X_fakeB_half, X_fakeB_half, X_fakeB_half], dim=1)
            

            optimizerG_f.zero_grad()
            optimizerG_c.zero_grad() 
            
            g_f_mse = 10*mse(X_fakeB, X_realB)
            d_f_hinge = hinge_loss(d_feat1_fake[0], y1)
            d_f_ca = 10*ca_loss(d_feat1_fake[1], y2)
            d_f_ef = ef_loss_changed(d_feat1_real[2], d_feat1_fake[2])
            g_f_hinge = 10*hinge_loss(X_fakeB, X_realB)
            g_f_vggloss = 10*vgg_loss(X_fakeB_stacked, X_realB_stacked)
            # gan1_loss = g_f_mse+d_f_hinge+d_f_ca+d_f_ef+g_f_hinge+g_f_vggloss
            gan1_loss = g_f_mse+d_f_hinge+d_f_ef+g_f_hinge+g_f_vggloss
            
            
                       
            g_c_mse = 10*mse(X_fakeB_half, X_realB_half)
            d_c_hinge = hinge_loss(d_feat2_fake[0], y1_half)
            d_c_ca = 10*ca_loss(d_feat2_fake[1], y2)
            d_c_ef = ef_loss_changed(d_feat2_real[2], d_feat2_fake[2])
            g_c_hinge = 10*hinge_loss(X_fakeB_half, X_realB_half)
            g_c_vggloss = 10*vgg_loss(X_fakeB_half_stacked, X_realB_half_stacked)
            
            # gan2_loss = g_c_mse+d_c_hinge+d_c_ca+d_c_ef+g_c_hinge+g_c_vggloss
            gan2_loss = g_c_mse+d_c_hinge+d_c_ef+g_c_hinge+g_c_vggloss
            
            gan_loss = gan1_loss + gan2_loss
            gan_loss.backward()
            optimizerG_f.step()
            optimizerG_c.step()

            with torch.no_grad():
                d_f_loss = d_loss1.item() + d_loss2.item()
                d_c_loss = d_loss3.item() + d_loss4.item()

                D_f_loss += d_f_loss
                D_c_loss += d_c_loss

                gan_loss = gan1_loss.item()+gan2_loss.item()
                Gan_loss += gan_loss
            
            if (i+1)%50==0:
                
                print()
                print(">%d<%d>: d_f_loss: %5f   d_c_loss: %5f   gan_loss: %5f" % (epoch+1, 1, d_f_loss, d_c_loss, gan_loss))
                # print(">%d<%d>: d_f_hinge: %5f, d_f_ca: %5f, d_f_ef: %5f, g_f_mse: %5f, g_f_hinge: %5f, g_f_vggloss: %5f" % (epoch+1, 2, d_f_hinge, d_f_ca, d_f_ef, g_f_mse, g_f_hinge, g_f_vggloss)) 
                print(">%d<%d>: d_f_hinge: %5f, d_f_ef: %5f, g_f_mse: %5f, g_f_hinge: %5f, g_f_vggloss: %5f" % (epoch+1, 2, d_f_hinge, d_f_ef, g_f_mse, g_f_hinge, g_f_vggloss)) 
                
                # print(">%d<%d>: d_c_hinge: %5f, d_c_ca: %5f, d_c_ef: %5f, g_c_mse: %5f, g_c_hinge: %5f, g_c_vggloss: %5f" % (epoch+1, 3, d_c_hinge, d_c_ca, d_c_ef, g_c_mse, g_c_hinge, g_c_vggloss)) 
                print(">%d<%d>: d_c_hinge: %5f, d_c_ef: %5f, g_c_mse: %5f, g_c_hinge: %5f, g_c_vggloss: %5f" % (epoch+1, 3, d_c_hinge, d_c_ef, g_c_mse, g_c_hinge, g_c_vggloss)) 
                print()
                
                viz.line([[d_f_loss],[d_c_loss], [gan_loss]], [count+i+1], win="VTGAN_LOSS", update="append")
                
            
            if (i+1)%40==0:    
                d_lr_decay_c.step()
                d_lr_decay_f.step() 
                
            
        D_f_loss /= len_along_epoch
        D_c_loss /= len_along_epoch
        Gan_loss /= len_along_epoch
        
        print(">>>>%d: d_f_loss: %5f   d_c_loss: %5f   gan_loss: %5f" % (epoch+1, D_f_loss, D_c_loss, Gan_loss))  
        iter_summarize_performance(g_model_fine, g_model_coarse, val_iter, str(epoch+1), True)
        iter_summarize_performance(g_model_fine, g_model_coarse, val_iter, str(epoch+1), False)  
        count += len_along_epoch    
            

            
           
            
            
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='/home/fzj/VT_normal/config/train_config.yaml')
    
    args = parser.parse_args()
    main(args)
    