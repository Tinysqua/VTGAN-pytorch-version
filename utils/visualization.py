from visdom import Visdom
import random
from os.path import join
from PIL import Image
from torchvision import transforms
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

viz_image = Visdom()
viz_half = Visdom()

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)

def summarize_performance(g_f_model, g_c_model, test_dir, iteration_str, is_half=False):
    g_f_model.eval()
    g_c_model.eval()
    ram_num = random.randint(1,30)
    ram_num = str(ram_num)
    test_fundus = join(test_dir, ram_num+".jpg")
    test_angio = join(test_dir, ram_num+"-"+ram_num+".jpg")
    X_realA = transformer(funloader(test_fundus)).unsqueeze(0).cuda()
    X_realB = transformer(angloader(test_angio)).unsqueeze(0).cuda()
    
    X_realA_half = transformer_resize(funloader(test_fundus)).unsqueeze(0).cuda()
    X_realB_half = transformer_resize(angloader(test_angio)).unsqueeze(0).cuda()
    
    X_fakeB_half, X_global = g_c_model(X_realA_half)
    X_fakeB = g_f_model(X_realA, X_global)
    
    
    
    if is_half:
        X_fakeB_half = (X_fakeB_half+1)/2
        X_fakeB_half = torch.cat([X_fakeB_half, X_fakeB_half, X_fakeB_half], dim=1)
        X_realB_half = (X_realB_half+1)/2
        X_realB_half = torch.cat([X_realB_half, X_realB_half, X_realB_half], dim=1)
        X_realA_half = (X_realA_half+1)/2
        display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0)
        display_list = display_list.detach().cpu().numpy()
        # display_list.append((X_realA_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_fakeB_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_realB_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        viz_image.images(display_list, env="VT_global", opts=dict(title= iteration_str))
        
    else:
        X_fakeB = (X_fakeB+1)/2
        X_fakeB = torch.cat([X_fakeB, X_fakeB, X_fakeB], dim=1)
        X_realB = (X_realB+1)/2
        X_realB = torch.cat([X_realB, X_realB, X_realB], dim=1)
        X_realA = (X_realA+1)/2
        display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0)
        display_list = display_list.detach().cpu().numpy()
        # display_list.append((X_realA*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_fakeB*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_realB*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        viz_image.images(display_list, env="VT_local", opts=dict(title= iteration_str))
    
  
    g_f_model.train()
    g_c_model.train()

def iter_summarize_performance(g_f_model, g_c_model, iter_thing, iteration_str, is_half=False):

    X_realA, X_realB, X_realA_half, X_realB_half, _ = next(iter_thing)
                
    X_realA = convert_to_cuda(X_realA)
    X_realB = convert_to_cuda(X_realB)
    X_realA_half = convert_to_cuda(X_realA_half)
    X_realB_half = convert_to_cuda(X_realB_half)
    
    X_fakeB_half, X_global = g_c_model(X_realA_half)
    X_fakeB = g_f_model(X_realA, X_global)
    
    if is_half:
        X_fakeB_half = (X_fakeB_half+1)/2
        X_fakeB_half = torch.cat([X_fakeB_half, X_fakeB_half, X_fakeB_half], dim=1)
        X_realB_half = (X_realB_half+1)/2
        X_realB_half = torch.cat([X_realB_half, X_realB_half, X_realB_half], dim=1)
        X_realA_half = (X_realA_half+1)/2
        display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0)
        display_list = display_list.detach().cpu().numpy()
        # display_list.append((X_realA_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_fakeB_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_realB_half*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        viz_image.images(display_list, env="VT_global", opts=dict(title= iteration_str))
        
    else:
        X_fakeB = (X_fakeB+1)/2
        X_fakeB = torch.cat([X_fakeB, X_fakeB, X_fakeB], dim=1)
        X_realB = (X_realB+1)/2
        X_realB = torch.cat([X_realB, X_realB, X_realB], dim=1)
        X_realA = (X_realA+1)/2
        display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0)
        display_list = display_list.detach().cpu().numpy()
        # display_list.append((X_realA*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_fakeB*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        # display_list.append((X_realB*0.5 + 0.5).squeeze(0).detach().cpu().numpy())
        viz_image.images(display_list, env="VT_local", opts=dict(title= iteration_str))

    
    

def funloader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def angloader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=0.5, std=0.5)
])

transformer_resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=0.5, std=0.5)
])