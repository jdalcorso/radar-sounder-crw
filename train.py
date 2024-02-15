from torch.nn.functional import normalize, softmax, cross_entropy
from torch import einsum, cat, flip, eye, bmm, manual_seed, permute, zeros, tensor, save, rand, diag, ones
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda import device_count
from torch.nn import DataParallel
from utils import create_model, create_dataset, pos_embed, show_A
import matplotlib.pyplot as plt
import argparse
import time
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Train', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 1, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(16,16), type=int)
    parser.add_argument('--seq_length', default=4, type=int)
    parser.add_argument('--overlap', default=(12,0), type=int)
    # Train
    parser.add_argument('--batch_size', default = 128, type=int)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--lr', default = 1E-3, type = int)
    parser.add_argument('--tau', default = 0.01, type = int)
    # Dev
    parser.add_argument('--pos_embed', default = True, type = bool)
    return parser

def main(args):
    print(args)
    # Model
    model = create_model(args.model, args.pos_embed)
    model = model.to('cuda')
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)

    # Dataset
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    tau = args.tau

    # Train
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_epoch = []
        for batch, seq in enumerate(dataloader):
            seq = seq.to('cuda')
            B, T, N, H, W = seq.shape   
            seq = seq.view(-1, H, W).unsqueeze(1) # BT x 1 x H x W
            if args.pos_embed:
                seq = pos_embed(seq)
            emb = model(seq).view(B, T, N, -1)  # B x T x N x C  # TODO QUESTO FORSE E' SBAGLIATO, C NON SI SPOSTA AUTOMATICAMENTE IN QUELLA POSIZIONE
            emb = normalize(emb, dim = -1) # L2 normalisation: now emb has L2norm=1 on C dimension
            emb = permute(emb, [0, 3, 1, 2])                                # B x C x T x N

            # Transition from t to t+1. We do a matrix product on the C dimension (i.e. computing cosine similarities)
            A = einsum('bctn,bctm->btnm', emb[:,:,:-1], emb[:,:,1:])/tau     # B x T-1 x N x N
            # TODO: set to zero except diagonals
            #mask = (diag(ones(N)) + diag(ones(N-1),1) + diag(ones(N-1),-1)).unsqueeze(0).unsqueeze(0).repeat(B,T-1,1,1).cuda()
            #A = mask * A
            #if batch == 0:
            #    show_A(A)
            
            # Transition energies for palindrome graphs. Sum of rows is STILL not 1. We dont have probabilities yet, we have cosine similarities
            AA = cat((A, flip(A,dims = [1]).transpose(-1,-2)), dim = 1)   # B x 2*T-2 x N x N
            #AA[rand([B, 2*T-2, N, N])< 0.2] = -1e10    # Edge Dropout (worsen performance)
            loss = 0

            for k in range(1,T-1):
                At = zeros(1,N,N, device = 'cuda')
                At[0,:,:] = eye(N)
                At = At.repeat([B,1,1]) # now At is B identity matrices stacked
                I = At
                #Do walk
                AA_this = cat([AA[:,:k],AA[:,-k:]], dim=1)
                for t in range(1,2*k):
                    current = AA_this[:,t] #+ eye(N, device = 'cuda').unsqueeze(0).repeat([B,1,1]) * args.diag_factor
                    At = bmm(softmax(current, dim = -1), At)
                loss += cross_entropy(input = At, target = I)

            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)

        print('Epoch:',epoch,'Loss:',loss_epoch.item(), 'Time:', time.time()-t0)

    plt.plot(loss_tot)
    plt.savefig('./crw/loss.png')
    plt.close()
    save(model.state_dict(), './crw/latest.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)