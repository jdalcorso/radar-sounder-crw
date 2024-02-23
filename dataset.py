import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset 

class RGDataset(Dataset):
    def __init__(self, filepath ='/data/MCoRDS1_2010_DC8/RG2_MCoRDS1_2010_DC8.pt', length = 10, dim = (24,24), overlap = (0,0), flip = False):
        self.filepath = filepath # 410 x 27330
        self.l = length
        self.T = torch.load(filepath)

        # Trim Miguel dataset
        if filepath.endswith('rg2.pt'):
            self.T = trim_miguel(self.T, length, dim)
            print('Trimmed Dataset to match radargram sizes!')

        if flip:
            self.T = torch.flip(input = self.T, dims = (1,))

        H, W = self.T.shape
        h, w = dim
        oh, ow = overlap
        self.nh = (H - oh)//(h - oh) # Formula is: N = (L-l)//(l-o)+1
        l = (self.l*(w-ow)+ow) # = self.pxw
        self.nw = (W - l)//(w-ow) + 1
        self.oh, self.ow = oh, ow
        self.h, self.w = h,w
        self.pxh = self.nh * h - oh*(self.nh-1)
        self.pxw = self.l * w - ow*(self.l-1)
        print('Total items:', self.nw, 'Length of item in pixels:', self.pxw)

    def __len__(self):
        return self.nw

    def __getitem__(self,index):
        item = self.T[:self.pxh,(self.w-self.ow)*index:(self.w-self.ow)*index+self.pxw]
        item = item.unfold(dimension = 0, size = self.h, step= self.h-self.oh)
        item = item.unfold(dimension = 1, size = self.w, step= self.w-self.ow)
        item = torch.permute(item, [1,0,2,3])
        return item.float()


if __name__ == '__main__':
    ds = RGDataset(filepath = 'datasets/MCORDS1_Miguel/rg2.pt', length = 4, dim = (32,32), overlap = (0,0))
    T, N, H, W = ds[0].shape
    images = ds[0]
    fig, axes = plt.subplots(N, T, figsize = (13,13))
    for t in range(T):
        for n in range(N):
            image = images[t, n].cpu().numpy()
            axes[n, t].imshow(image, cmap='gray')
            axes[n, t].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('grid.png')
    plt.close()
    print('Main done.')

def trim_miguel(T, length, dim):
    splits = torch.tensor([9984, 6656, 9984, 20000, 16640, 32864, 8992])
    splits_cum = torch.cumsum(splits, dim = 0)
    new_T = []
    for i in range(splits.shape[0]):
        # check length of radargram
        idx_start = 0 if i == 0 else splits_cum[i-1]
        L = splits[i]
        nrgs = torch.floor(L/(dim[1]*length)).long()
        effective_length = nrgs * (dim[1]*length)
        slice_of_T = T[:,idx_start:idx_start+effective_length]
        
        new_T.append(slice_of_T)
    T = torch.cat(new_T, dim = 1)
    return T