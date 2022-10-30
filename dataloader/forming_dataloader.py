from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob
from os.path import join

def find_other_name(be_found):
    filename = be_found.split("/")[-1].split(".")[0]
    left, right = filename.split("_")[0], filename.split("_")[1]
    return left+"_mask_"+right+".png"

def get_address_list(up_dir, picture_form):
    return glob.glob(up_dir+'*.'+picture_form)

def convert_to_resize(X):
    y1 = transformer(X)
    y2 = transformer_resize(X)
    return y1, y2

class Fundus_angio_dataset(data.Dataset):
    def __init__(self, imgs_path):
        #imgs_path is the directory that above the 'ab' and the 'nor' directory
        super(Fundus_angio_dataset, self).__init__()
        ab_path = join(imgs_path, "ab")
        nor_path = join(imgs_path, "nor")
        self.ab_fundus = get_address_list(join(ab_path, "Images/"),"png")
        # self.ab_angio = get_address_list(join(ab_path, "Masks/"), "png")
        self.ab_angio = join(ab_path, "Masks/")
        self.nor_fundus = get_address_list(join(nor_path, "Images/"), "png")
        # self.nor_angio = get_address_list(join(nor_path, "Masks/"), "png")
        self.nor_angio = join(nor_path, "Masks/")
    def __getitem__(self, index):
        ab_f = self.funloader(self.ab_fundus[index])
        ab_f, ab_f_half = convert_to_resize(ab_f)
        
        nor_f = self.funloader(self.nor_fundus[index])
        nor_f, nor_f_half = convert_to_resize(nor_f)
        
        ab_an = self.angloader(self.ab_angio + find_other_name(self.ab_fundus[index]))
        ab_an, ab_an_half = convert_to_resize(ab_an)
        
        nor_an = self.angloader(self.nor_angio + find_other_name(self.nor_fundus[index]))
        nor_an, nor_an_half = convert_to_resize(nor_an)
        
        ab_data = (ab_f, ab_an, ab_f_half, ab_an_half, False)
        nor_data = (nor_f, nor_an, nor_f_half, nor_an_half, True)
        
        #since the quatities of data, I return a list. Don't forget to randomly choose the data
        return [ab_data, nor_data]
        
    def __len__(self):
        return len(self.ab_fundus)
        
    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def angloader(self, path):
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