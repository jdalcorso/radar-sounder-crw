import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset 

class MCORDS1Dataset(Dataset):
    def __init__(self, filepath ='/data/MCoRDS1_2010_DC8/RG_MCoRDS1_2010_DC8.pt', length = 10, dim = (24,24), overlap = (0,0)):
        self.filepath = filepath # 410 x 27350
        self.items = []
        l = length
        T = torch.load(filepath)
        H, W = T.shape
        h, w = dim[0], dim[1]
        oh,ow = overlap[0], overlap[1]
        nh, nw = H//(h - oh)-1, W//(w- ow)-1
        columns = []

        # Create columns of overlapping patches
        for i in range(nw):
            column  = torch.zeros(nh, h, w)
            for j in range(nh):
                column[j,:,:] = T[j*(h-oh):j*(h-oh)+h, i*(w-ow):i*(w-ow)+w]
            columns.append(column)

        # Create items of the dataset as sequences of dim TxNxhxw
        len_dataset = len(columns)//length
        for i in range(len_dataset):
            self.items.append(torch.stack(columns[i*l:i*l+l], dim = 0))
        #for i in range(len(columns)-l):
        #    self.items.append(torch.stack(columns[i:i+l], dim = 0))

        print('Total items:', len(self.items), ', Dim items:', self.items[0].shape)

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        return self.items[index].float()

if __name__ == '__main__':
    ds = MCORDS1Dataset(length = 10, dim = (24,24), overlap = (12,12))
    print('Shape', ds[1].shape)

    # Assuming you have a tensor with shape (T, N, H, W)
    T, N, H, W = ds[1].shape
    images = ds[1]

    # Plotting all TxN images in a grid
    fig, axes = plt.subplots(N, T, figsize = (13,13))

    for t in range(T):
        for n in range(N):
            # Get the image at time step t and node n
            image = images[t, n].cpu().numpy()

            # Display the image in the corresponding subplot
            axes[n, t].imshow(image, cmap='gray')  # Assuming grayscale images
            axes[n, t].axis('off')  # Turn off axis labels for clarity

    plt.tight_layout()

    plt.show()
    plt.tight_layout()
    plt.savefig('grid.png')
    plt.close()