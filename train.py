from torch import manual_seed, tensor, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda import device_count
from torch.nn import DataParallel
from utils import create_model, create_dataset
from model import CRW
import matplotlib.pyplot as plt
import argparse
import time
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Train', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 0, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=4, type=int)
    parser.add_argument('--overlap', default=(16,0), type=int)
    # Train
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--lr', default = 1E-3, type = int)
    parser.add_argument('--tau', default = 0.01, type = int)
    # Dev
    parser.add_argument('--pos_embed', default = True, type = bool)
    parser.add_argument('--dataset_full', default = False)
    return parser

def main(args):
    print(args)
    # Model
    encoder = create_model(args.model, args.pos_embed)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    model = CRW(encoder, args.tau, args.pos_embed)
    model = model.to('cuda')

    # Dataset
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, full = args.dataset_full, overlap = args.overlap)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    # Train
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_epoch = []
        for batch, seq in enumerate(dataloader):
            seq = seq.to('cuda')
            loss = model(seq)
            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)

        print('Epoch:',epoch,'Loss:',loss_epoch.item(), 'Time:', time.time()-t0)

    plt.plot(loss_tot)
    plt.savefig('./crw/output/a_loss.png')
    plt.close()
    save(encoder.state_dict(), './crw/latest.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)