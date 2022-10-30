from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob
from os.path import join

def get_address_list(up_dir, picture_form):
    return glob.glob(up_dir+'*.'+picture_form)

def convert_to_resize(X):
    y1 = transformer(X)
    y2 = transformer_resize(X)
    return y1, y2

class Official_Fundus_angio_dataset(data.Dataset):
    def __init__(self, up_dir):
        super(Official_Fundus_angio_dataset, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path =  get_address_list(fu_path, "png")
        
    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        if first_num<8:
            is_nor = False
        else:
            is_nor = True
        XReal_A, XReal_A_half = convert_to_resize(self.funloader(fun_filename))
        an_filename = str(first_num)+"_mask_"+str(second_num)+".png"
        an_file_path = self.an_path + an_filename
        XReal_B, XReal_B_half = convert_to_resize(self.angloader(an_file_path))
        
        return XReal_A, XReal_B, XReal_A_half, XReal_B_half, is_nor
    
    def __len__(self):
        return len(self.fu_path)
    
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

