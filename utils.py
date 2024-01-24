from torch import load, tensor
from scipy.io import loadmat
from model import CNN, Resnet
from dataset import MCORDS1Dataset, MiguelDataset

def create_model(id):
    if id == 0:
        return CNN()
    if id == 1:
        return Resnet()
    
def create_dataset(id, length, dim, overlap):
    if id == 0:
        return MCORDS1Dataset(length = length, dim = dim, overlap = overlap)
    if id == 1:
        return MiguelDataset(length = length, dim = dim, overlap = overlap)
    
def get_reference(id,h,w):
    # Returns number of classes and reference initial segmentation
    if id == 0:
        return 4, load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')[:h,:w]
    if id == 1:
        seg = loadmat('./datasets/MCORDS1_Miguel/gt/gt_01.mat')
        return 4, tensor(seg['gtR1_v3'])[:h,:w]
