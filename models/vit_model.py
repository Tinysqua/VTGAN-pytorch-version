import torch.nn.functional as F
import torch
from einops import rearrange
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEncoder(torch.nn.Module):
    def __init__(self, num_patches=64, projection_dim=64):
        super(PatchEncoder, self).__init__()
        self.num_patches= num_patches
        self.projection = torch.nn.Linear(64*64*4,out_features=projection_dim)#256*256:262144
        self.projection_resize = torch.nn.Linear(32*32*4, out_features=projection_dim)
        self.position_embedding = torch.nn.Embedding(num_patches,projection_dim)

    def forward(self, input):
        
        # positions = torch.nn.Parameter(torch.randn(1, self.num_patches, self.num_patches)).cuda()
        positions = torch.arange(self.num_patches).cuda()
        if input.shape[-1]==4096:
            encoded = self.projection_resize(input)+self.position_embedding(positions)
        else:
             encoded = self.projection(input) + self.position_embedding(positions)
        return encoded
        # if input.shape[-1]==4096:
        #     encoded = self.projection_resize(input)+self.position_embedding(positions)
        # else:
        #      encoded = self.projection(input) + self.position_embedding(positions)
        # return encoded
        
class Block(torch.nn.Module):
    def __init__(self, project_dim, depth, num_heads, mlp_ratio):
        super(Block, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.normlayer1 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.normlayer2 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.linear1 = torch.nn.Linear(project_dim, project_dim * mlp_ratio)
        self.linear2 = torch.nn.Linear(project_dim * mlp_ratio, project_dim)
        self.gelu = torch.nn.GELU()
        for i in range(depth):
            setattr(self, "layer"+str(i+1), torch.nn.MultiheadAttention(project_dim, num_heads, dropout=0.1))
    
    def forward(self, encoded_patches):
        feat = []
        for i in range(self.depth):
            x1 = self.normlayer1(encoded_patches)
            attention_output, attn_output_weights = getattr(self, "layer"+str(i+1))(x1, x1, x1)
            x2 = encoded_patches + attention_output
            x3 = self.normlayer2(x2)
            x3 = self.mlp(x3)
            encoded_patches = x2 + x3
            feat.append(encoded_patches)
        feat_total = torch.cat([feat[0], feat[1], feat[2], feat[3]], -1)
        return feat_total, encoded_patches
            
            
            
    def mlp(self, x, dropout_rate=0.1):
        x = self.linear1(x)
        x = self.gelu(x)
        x = F.dropout(x, p=dropout_rate)
        x = self.linear2(x)
        x = self.gelu(x)
        x = F.dropout(x, p=dropout_rate)
        return x
    

# patchsize: 64 
class vit_discriminator(torch.nn.Module):
    def __init__(self, patch_size, project_dim=64,num_heads=4, mlp_ratio=2, depth=4):
        super(vit_discriminator, self).__init__()
        self.patch_size = patch_size
        self.GELU = torch.nn.GELU()
        self.block = Block(project_dim=project_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio).cuda()
        self.Conv_4_1 = torch.nn.Conv2d(1, 1, (4, 4), padding='same')
        self.Conv_4_1_2 = torch.nn.Conv2d(1, 64, (4,4), padding='same')
        # self.Conv_4_1 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        # self.Conv_4_1_2 = torch.nn.Conv2d(1, 64, (3,3), padding=1)
        self.MultiHeadAttention = torch.nn.MultiheadAttention(project_dim, num_heads, dropout=0.1)
        #self.LayerNorm = torch.nn.LayerNorm(64, eps=1e-6)
        self.linear3 = torch.nn.Linear(64, 2)
        self.Softmax = torch.nn.Softmax(dim=-1)
        self.AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        self.num_patches = (512 // 64) ** 2
        self.PatchEncoder=PatchEncoder(projection_dim=project_dim).cuda()
        self.LayerNormalization_0 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.LayerNormalization_1 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.LayerNormalization_2 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)


    def forward(self,fundus,angio):
        patch_size = self.patch_size
        X = torch.cat((fundus, angio), 1)
        feat = []
        patches = rearrange(X, 'b c (h h1) (w w2) -> b (h w) (h1 w2 c)', h1=patch_size, w2=patch_size)
        encoded_patches = self.PatchEncoder(patches)
        
        feat, encoded_patches = self.block(encoded_patches)

        representation = self.LayerNormalization_0(encoded_patches)
        
        X_reshape = representation.unsqueeze(0).permute(1,0,2,3)
        X = self.Conv_4_1(X_reshape)
        out_hinge = torch.tanh(X)
        representation = self.Conv_4_1_2(X_reshape)
        features = self.AdaptiveAvgPool2d(representation).squeeze(-1).squeeze(-1)
        classses = self.linear3(features)
        out_class = self.Softmax(classses)
        return [out_hinge, out_class, feat]

