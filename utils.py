from torch import load, arange, cat, argmax, einsum, tensor
from torch.utils.data import Subset
from encoder import CNN, Resnet
from dataset import RGDataset, trim_miguel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
import ruptures as rpt
from matplotlib.colors import ListedColormap
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.nn.functional import softmax, normalize, cross_entropy
import numpy as np

def create_model(id, pos_embed):
    if id == 0:
        return CNN(pos_embed)
    if id == 1:
        return Resnet(pos_embed)
    
def create_dataset(id, length, dim, overlap, full = False, flip = False):
    if id == 0:
        ds = RGDataset(filepath ='/data/MCoRDS1_2010_DC8/RG2_MCoRDS1_2010_DC8.pt', length = length, dim = dim, overlap = overlap, flip = flip)
    if id == 1:
        ds = RGDataset(filepath = 'datasets/MCORDS1_Miguel/rg2.pt', length = length, dim = dim, overlap = overlap, flip = flip)
    if full:
        return ds
    else:
        non_overlapping_idx = range(0,len(ds), length)
        print('Non-Overlapping dataset! Number of items is the above divided by the length of the sequence...')
        return Subset(ds, non_overlapping_idx)



def get_reference(id,h,w, flip = False, length = None, dim = None, overlap = None):
    # Returns number of classes and reference initial segmentation
    # w = 0 -> return the whole dataset 
    if id == 0:
        data = load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')
        nclasses = 4
    # GT only for the first radargram
    if id == 1:
        data = load('./datasets/MCORDS1_Miguel/seg2.pt')
        data = trim_miguel(data, length, dim)
        nclasses = 6
    # Same as id==0 but with uncertain class
    if id == 2:
        data = load('/data/MCoRDS1_2010_DC8/SG3_MCoRDS1_2010_DC8.pt')
        nclasses = 4
    data = data[:h,:] if w==0 else data[:h,:w]
    return (nclasses, torch.flip(data, (1,))) if flip else (nclasses, data)


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

@torch.no_grad()
def propagate(seq, seg_ref, model, lp, nclasses, do_pos_embed, use_last):
    '''
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
    I = ndiag_matrix(N, 3).cuda()
    xent = torch.zeros(N,T-1, requires_grad = False)
    for i in range(T-1):
        At = A[i,:,:]
        xent[:,i] = (cross_entropy(input = At, target = I, reduction='none'))
    xent = xent.unfold(dimension = 1, step = 1, size = 10).detach()
    xent = xent.mean(dim=(-1,-2))
    pelt = rpt.Pelt(model="rbf").fit(xent)
    result = pelt.predict(pen=10)
    change_idx = result[0] - 10
    change_idx = torch.maximum(torch.tensor(0),torch.tensor(change_idx)).item()

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
    # Create a zero tensor with the desired size (n <= 2 is id, n = 3 is tri, n = 4 is penta)
    matrix = torch.zeros(size, size)
    matrix.diagonal(offset=0).fill_(1)
    for i in range(0,n-1):
        matrix.diagonal(offset=i).fill_(1)
        matrix.diagonal(offset=-i).fill_(1)
    matrix = matrix/matrix.sum(dim=1).unsqueeze(0).transpose(0,1)
    return matrix

def rolling_variance(image, window_size):
    #unfolded = torch.nn.functional.unfold(image.unsqueeze(0), kernel_size=(1, window_size), stride=(1, 1))
    unfolded = image.unfold(dimension = -1, size = window_size, step = 1)
    unfolded = torch.permute(unfolded, [0, 2, 1])
    mean = torch.mean(unfolded, dim = (0,1), keepdim=True)
    squared_diff = (unfolded - mean)**2
    variance = torch.mean(squared_diff, dim = (0,1))
    return variance

def plot_kmeans(emb, T, N):
    # show KMeans on features
    kmeans = KMeans(4, n_init = 'auto')
    kmeans_fitted = kmeans.fit(emb[0].reshape(-1,128).cpu().detach())
    plt.imshow(tensor(kmeans_fitted.labels_).view(T,N).transpose(0,1))
    plt.savefig('./crw/output/_kmeans.png')
    plt.axis('off')
    plt.close()

def plot(img, save = None, seg = None):
    if seg is None:
        plt.imshow(img, interpolation="nearest", cmap = 'gray')
        plt.tight_layout()
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.close()
    else:
        plt.figure(figsize = (13,13))
        plt.subplot(211)
        fs = 12
        #colors = [(0,0,0), (0.33,0.33,0.33), (1,0,0), (1,1,1)] # for MCORDS1
        class_colors = {
            0: (0,0,0),
            1: (1,1,1),
            2: 'red',
            3: (0.33,0.33,0.33),
            4: (0.66,0.66,0.66),
            5: (1,1,1)
        }
        cmap = ListedColormap([class_colors[i] for i in range(6)])
        #cmap = ListedColormap(colors)
        plt.imshow(img, interpolation="nearest", cmap = cmap, vmin = 0, vmax = 5)
        num_ticks = 5
        new_y_ticks = np.linspace(0, 1256, num_ticks)  # 410 for MCORDS1, 1256 for MCORDS3
        new_y_labels = [f'{i*0.103:.2f}' for i in range(num_ticks)] # 0.103us is the timestep for MCORDS1       
        plt.yticks(new_y_ticks, new_y_labels,fontsize = fs)
        plt.ylabel('Time [μs]',fontsize = fs)
        plt.xlabel('Trace',fontsize = fs)
        plt.subplot(212)
        plt.imshow(seg, cmap = cmap, interpolation="nearest", vmin = 0, vmax = 5)
        plt.yticks(new_y_ticks, new_y_labels, fontsize = fs)
        plt.ylabel('Time [μs]',fontsize = fs)
        plt.xlabel('Trace',fontsize = fs)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        plt.close()
