from torch import load, arange, cat, argmax
from encoder import CNN, Resnet
from dataset import MCORDS1Dataset, MiguelDataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.nn.functional import softmax, normalize

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
    
def get_reference(id,h,w, flip = False):
    # Returns number of classes and reference initial segmentation
    # w = 0 -> return the whole dataset 
    if id == 0:
        data = load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')
        nclasses = 4
    # GT only for the first radargram
    if id == 1:
        data = load('./datasets/MCORDS1_Miguel/seg2.pt')
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

def plot(img, save = None, seg = None):
    if seg is None:
        plt.imshow(img, interpolation="nearest")
        plt.tight_layout()
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.close()
    else:
        plt.subplot(121)
        plt.imshow(img, interpolation="nearest")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(seg)
        plt.axis('off')
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        plt.close()

def propagate(seq, t, seg, model, lp, nclasses, rg_len, do_pos_embed, use_last):
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
    seq = seq.view(-1, H, W).unsqueeze(1)
    if do_pos_embed:
        seq = pos_embed(seq)
    emb = model(seq).view(T,N,-1)
    emb = normalize(emb, dim = -1) # L2

    feats = []
    masks = []

    final_prediction = torch.zeros(N,T, device = 'cuda')

    # Add reference mask and features
    seg_ref = seg[:,rg_len * t:rg_len * t + W]

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
    
    return final_prediction

def ndiag_matrix(size, n = 1):
    # Create a zero tensor with the desired size (n <= 2 is id, n = 3 is tri, n = 4 is penta)
    matrix = torch.zeros(size, size)
    matrix.diagonal(offset=0).fill_(1)
    for i in range(0,n-1):
        matrix.diagonal(offset=i).fill_(1)
        matrix.diagonal(offset=-i).fill_(1)
    matrix = matrix/matrix.sum(dim=1).unsqueeze(0).transpose(0,1)
    return matrix
    return matrix