from torch.nn.functional import normalize, softmax, cross_entropy
from torch import einsum, cat, flip, eye, bmm, manual_seed, permute, zeros, tensor, clone, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda import device_count
from torch.nn import DataParallel
from model import CNN, Resnet
from dataset import MCORDS1Dataset
import matplotlib.pyplot as plt
import argparse
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Train', add_help=False)
    # Data
    parser.add_argument('--patch_size', default=(12,12), type=int)
    parser.add_argument('--seq_length', default=10, type=int)
    parser.add_argument('--overlap', default=(0,0), type=int)
    # Train
    parser.add_argument('--batch_size', default = 64, type=int)
    parser.add_argument('--epochs', default = 20, type = int)
    parser.add_argument('--tau', default = 0.01, type = int)
    return parser

def main(args):
    # Model
    model = Resnet()
    model = model.to('cuda')
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)

    # Dataset
    dataset = MCORDS1Dataset(length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = args.epochs
    tau = args.tau

    # Train
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        loss_epoch = []
        for batch, seq in enumerate(dataloader):
            seq = seq.to('cuda')
            B, T, N, H, W = seq.shape    
            emb = model(seq.view(-1, H, W).unsqueeze(1)).view(B, T, N, -1)  # B x T x N x C  # TODO QUESTO FORSE E' SBAGLIATO, C NON SI SPOSTA AUTOMATICAMENTE IN QUELLA POSIZIONE
            emb = normalize(emb, dim = -1) # L2 normalisation: now emb has L2norm=1 on C dimension
            emb = permute(emb, [0, 3, 1, 2])                                # B x C x T x N

            # Transition from t to t+1. We do a matrix product on the C dimension (i.e. computing cosine similarities)
            A = einsum('bctn,bctm->btnm', emb[:,:,:-1], emb[:,:,1:])/tau     # B x T-1 x N x N
            # Transition energies for palindrome graphs. Sum of rows is STILL not 1. We dont have probabilities yet, we have cosine similarities
            AA = cat((A, flip(A,dims = [1]).transpose(-1,-2)), dim = 1)   # B x 2*T-2 x N x N
            #AA[rand([B, 2*T-2, N, N])< 0.1] = -1e10    # Edge Dropout
            loss = 0

            # For each of the k palindrome paths
            for k in range(1):
                At = zeros(1,N,N, device = 'cuda')
                At[0,:,:] = eye(N)
                At = At.repeat([B,1,1]) # now At is B identity matrices stacked
                if k == 0:
                    I = clone(At)
                # Do walk
                for t in range(k, 2*T-2-k):
                    At = bmm(softmax(AA[:,t], dim = -1), At)
                loss += cross_entropy(input = At, target = I)

            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)

        print('Epoch:',epoch,'Loss:',loss_epoch.item())

    plt.plot(loss_tot)
    plt.savefig('./crw/loss.png')
    plt.close()
    save(model.state_dict(), './crw/latest.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)