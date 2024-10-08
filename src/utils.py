import matplotlib.pyplot as plt
import torch
import numpy as np
import ruptures as rpt
from torch import load, arange, cat, argmax, einsum
from torch.utils.data import Subset
from encoder import CNN, Resnet
from dataset import RGDataset, trim_miguel
from matplotlib.colors import ListedColormap
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.nn.functional import normalize, cross_entropy


def create_model(id, pos_embed):
    '''
    Chooses between custom CNN and Resnet
    '''
    if id == 0:
        return CNN(pos_embed)
    if id == 1:
        return Resnet(pos_embed)


def create_dataset(id, length, dim, overlap, full = False, flip = False):
    '''
    Chooses the dataset between the ones used in the paper. Modify this snippet to load a custom dataset.
    A new dataset would be a new .pt file of arbitrary dimension HxW (i.e. 1 channel radargram)
    '''
    # MCORDS1
    if id == 0:
        ds = RGDataset(filepath ='/data/MCoRDS1_2010_DC8/RG2_MCoRDS1_2010_DC8.pt', length = length, dim = dim, overlap = overlap, flip = flip)
    # MCORDS3
    if id == 1:
        ds = RGDataset(filepath = '/datasets/MCORDS1_Miguel/rg2.pt', length = length, dim = dim, overlap = overlap, flip = flip)
    # SHARAD
    if id == 3:
        ds = RGDataset(filepath = '/datasets/SHARAD/sharad_north_rg.pt', length = length, dim = dim, overlap = overlap, flip = flip)
    if full:
        return ds
    else:
        non_overlapping_idx = range(0,len(ds), length)
        print('Non-Overlapping dataset! Number of items is the above divided by the length of the sequence...')
        return Subset(ds, non_overlapping_idx)


def get_reference(id,h,w, flip = False, length = None, dim = None, overlap = None):
    '''
    Gets the reference of the corresponding dataset. Used when performing inference.
    Modify this to do inference on a personalized dataset.
    The segmentation map should match the size of the dataset HxW and contain class indices
    as integers starting from 0.
    '''
    # Returns number of classes and reference initial segmentation
    # w = 0 -> return the whole dataset 
    if id == 0:
        data = load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')
        nclasses = 4
    # GT only for the first radargram
    if id == 1:
        data = load('/datasets/MCORDS1_Miguel/seg3.pt') # Shifted segmentation
        data = trim_miguel(data, length, dim)
        nclasses = 6
    # Same as id==0 but with uncertain class
    if id == 2:
        data = load('/data/MCoRDS1_2010_DC8/SG3_MCoRDS1_2010_DC8.pt')
        nclasses = 4
    # SHARAD dataset
    if id == 3:
        data = load('/datasets/SHARAD/sharad_north_sg5.pt')
        nclasses = 5
    data = data[:h,:] if w==0 else data[:h,:w]
    return (nclasses, torch.flip(data, (1,))) if flip else (nclasses, data)


def pos_embed(seq):
    '''
    Adds a positional embedding channel to the input radargram.
    The usage of this function is triggered by an argument in the code and
    this also changes the input channels of the model from 1 to 2.
    Keep in mind that if you train a model with positional embedding you
    should also set pos_embed=True during inference/test to load the correct
    weights.
    '''
    # seq has size BT x 1 x H x W
    BT, _, H, W = seq.size()
    pe = arange(0, H).unsqueeze(-1)/H-0.5  # H x 1
    pe = pe.repeat([1,W]) # H x W
    pe = pe.unsqueeze(0).unsqueeze(0).repeat([BT,1,1,1]) # BT x 1 x H x W
    return cat([pe.to('cuda'),seq], dim = 1)


@torch.no_grad()
def propagate(seq, seg_ref, model, lp, nclasses, do_pos_embed, use_last):
    '''
    KNN Label propagation pipeline as per many Video Object Segmentation works.
    seq:        sequence of shape T, N, H, W
    t:          number of cycle within the seg (i.e. number of radargram)
    seg:        segmentation of shape H, W (H,W different from the above)
    model:      model trained with crw
    lp:         label propagation method
    nclasses:   number of classes
    rg_len:     length of the radargram (=N*W if no overlap in W dimension)
    pos_embed:  whether or not positional embedding has been used in the model
    use_last:   whether or not the last sample is used as reference
    '''
    T, N, H, W = seq.shape
    if use_last: seq = torch.flip(seq,(0,))

    # Obtain embeddings and reference mask
    seq = seq.reshape(-1, H, W).unsqueeze(1)
    if do_pos_embed:
        seq = pos_embed(seq)
    emb = model(seq).view(T,N,-1)
    emb = normalize(emb, dim = -1) # L2

    # Compute horizontality metric
    A = einsum('tnc,tmc->tnm', emb[:,:,:-1], emb[:,:,1:])/0.1
    I = ndiag_matrix(N,1).cuda() 
    xent = torch.zeros(N,T-1, requires_grad = False)
    for i in range(T-1):
        At = A[i,:,:]
        xent[:,i] = (cross_entropy(input = torch.transpose(At,0,1), target = I, reduction='none'))

    column_diffs = torch.tensor([torch.sum(torch.abs(xent[:, i] - xent[:, i+1])) for i in range(xent.shape[1] - 1)])
    try:
        pelt = rpt.Pelt(model="rbf").fit(column_diffs)
        result = pelt.predict(pen=5)
        change_idx = result[-2]+5
        change_idx = torch.maximum(torch.tensor(0),torch.tensor(change_idx)).item()
    except:
        change_idx=None

    feats = []
    masks = []

    final_prediction = torch.zeros(N,T, device = 'cuda')

    down = transforms.Resize((N,1), interpolation = InterpolationMode.NEAREST)

    label = down(seg_ref.unsqueeze(0)).squeeze(0).to('cuda')
    final_prediction[:,0] = label.squeeze(1)
    mask = torch.zeros(nclasses, N, 1, device = 'cuda')
    for class_idx in range(0, nclasses):
        m = (label == class_idx).unsqueeze(0).float()
        mask[class_idx, :, :] = m
    mask = mask.unsqueeze(0)
    feat = torch.permute(emb[0,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
    feats.append(feat)
    masks.append(mask)

    for n in range(1,T):
        feat = torch.permute(emb[n,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
        mask = lp.predict(feats = feats, masks = masks, curr_feat = feat)

        feats.append(feat)
        masks.append(mask)

        # Add new mask (no more one-hot) to the final prediction
        final_prediction[:,n] = argmax(mask, dim = 1).squeeze()
    return final_prediction, xent, change_idx


def ndiag_matrix(size, n = 1):
    '''
    Creates a k-diagonal matrix. May be useful to mask transitions.
    '''
    # Create a zero tensor with the desired size (n <= 2 is id, n = 3 is tri, n = 4 is penta)
    matrix = torch.zeros(size, size)
    matrix.diagonal(offset=0).fill_(1)
    for i in range(0,n-1):
        matrix.diagonal(offset=i).fill_(1)
        matrix.diagonal(offset=-i).fill_(1)
    matrix = matrix/matrix.sum(dim=1).unsqueeze(0).transpose(0,1)
    return matrix


def plot(img, save = None, seg = None, dataset = 0, aspect = 1):
    '''
    Plots with colormap according to the paper Experiments section.
    '''
    if dataset == 0:
        colors = [(0,0,0), (0.33,0.33,0.33), (1,0,0), (1,1,1)] # for MCORDS1
    if dataset == 1:
        colors = [
            (0,0,0,1), # black, free space
            (1,1,1,1), # white, noise
            (1,0,0,1), # red, bedrock
            (0.33,0.33,0.33,1), # dark gray, inland ice
            (0.66,0.66,0.66,1), # light gray, floating ice
            ]
    if dataset == 3:
        colors = [
            (0,0,0,1), # black, free space
            (0.33,0.33,0.33,1), # white, noise
            (1,0,0,1), # red, bedrock
            (1,1,1,1), # dark gray, inland ice
            (0.66,0.66,0.66,1), # light gray, floating ice
            ]
    cmap = ListedColormap(colors)
    if seg is None:
        plt.imshow(img, interpolation="nearest", cmap = cmap, vmin = 0, vmax = 4)
        plt.gca().set_aspect(aspect)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        plt.close()
    else:
        plt.figure(figsize = (13,13))
        plt.subplot(211)
        fs = 12
        plt.imshow(img, interpolation="nearest", cmap = cmap, vmin = 0, vmax = 4)
        plt.xlabel('Trace',fontsize = fs)
        plt.subplot(212)
        plt.imshow(seg, cmap = cmap, interpolation="nearest", vmin = 0, vmax = 4)
        plt.ylabel('Time [μs]',fontsize = fs)
        plt.xlabel('Trace',fontsize = fs)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        plt.close()
