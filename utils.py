from torch import load, tensor, arange, cat
from scipy.io import loadmat
from model import CNN, Resnet
from dataset import MCORDS1Dataset, MiguelDataset
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

def create_model(id, pos_embed):
    if id == 0:
        return CNN(pos_embed)
    if id == 1:
        return Resnet(pos_embed)
    
def create_dataset(id, length, dim, overlap, flip = False):
    if id == 0:
        return MCORDS1Dataset(length = length, dim = dim, overlap = overlap, flip = flip)
    if id == 1:
        return MiguelDataset(length = length, dim = dim, overlap = overlap, flip = flip)
    
def get_reference(id,h,w):
    # Returns number of classes and reference initial segmentation
    # w = 0 -> return the whole dataset 
    if id == 0:
        if w == 0:
            return 4, load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')[:h,:]
        else:
            return 4, load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')[:h,:w]
    # GT only for the first radargram
    if id == 1:
        seg = loadmat('./datasets/MCORDS1_Miguel/gt/gt_01.mat')
        if w == 0:
            return 4, tensor(seg['gtR1_v3'])[:h,:]
        else:
            return 4, tensor(seg['gtR1_v3'])[:h,:w]
    # Same as id==0 but with uncertain class
    if id == 2:
        if w == 0:
            return 4, load('/data/MCoRDS1_2010_DC8/SG3_MCoRDS1_2010_DC8.pt')[:h,:]
        else:
            return 4, load('/data/MCoRDS1_2010_DC8/SG3_MCoRDS1_2010_DC8.pt')[:h,:w]

def pos_embed(seq):
    # seq has size BT x 1 x H x W
    BT, _, H, W = seq.size()
    pe = arange(0, H).unsqueeze(-1)/H-0.5  # H x 1
    pe = pe.repeat([1,W]) # H x W
    pe = pe.unsqueeze(0).unsqueeze(0).repeat([BT,1,1,1]) # BT x 1 x H x W
    return cat([pe.to('cuda'),seq], dim = 1)

def show_A(A):
    # A with dimension B T-1 N N
    B, T, _, _ = A.shape    
    _, axes = plt.subplots(B, T, figsize=(T * 3, B * 3))
    axes = axes.flatten()
    
    for i in range(B * T):
        image = A[i // T, i % T].cpu().detach()  # Convert to NumPy array for Matplotlib
        axes[i].imshow(softmax(image,dim = 1), cmap='gray')  # Assuming grayscale images, adjust cmap if using color images
        axes[i].axis('off')  # Turn off axis labels
    
    plt.tight_layout()
    plt.show()
    plt.savefig('transitions.png')
    plt.close()
