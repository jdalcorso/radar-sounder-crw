from torch import manual_seed, tensor, load, permute, zeros, exp, cat
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.cuda import device_count
from torch.nn import DataParallel
from unet import UNet
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import argparse
import time
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('UNet train and test on SHARAD dataset', add_help=False)
    # Data
    parser.add_argument('--patch_size', default=(912,64), type=int)
    parser.add_argument('--split', default=0.9, type=float)
    # Train
    parser.add_argument('--batch_size', default = 64, type=int)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--lr', default = 1E-4, type = float)
    return parser

def main(args):
    print(args)
    # Model
    model = UNet(n_channels = 1, n_classes = 5, bilinear=True)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to('cuda')

    # Dataset (SHARAD)
    rg = load('/datasets/SHARAD/sharad_north_rg.pt')
    rg = permute(rg.unfold(dimension = 1, size = args.patch_size[1], step= args.patch_size[1]),[1,0,2])
    sg = load('/datasets/SHARAD/sharad_north_sg5.pt')
    sg_oh = zeros(5, sg.shape[0], sg.shape[1])
    sg_oh = sg_oh.scatter_(0, sg.unsqueeze(0).long(), 1)
    sg_oh = permute(sg_oh.unfold(dimension = 2, size = args.patch_size[1], step= args.patch_size[1]),[2,0,1,3])
    dataset = TensorDataset(rg,sg_oh)

    # Split
    train_ratio, test_ratio = args.split, 1-args.split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size   
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    # Train
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_epoch = []
        for batch, (sample, ref) in enumerate(train_loader):
            sample, ref = sample.to('cuda').unsqueeze(1), ref.to('cuda')
            ref = ref.to('cuda') 
            pred = model(sample)
            pred = F.softmax(pred, dim=1) # logits to prob
            loss = F.cross_entropy(input = pred , target = ref)
            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)
        print('Epoch:',epoch+1,'Loss:',loss_epoch.item(), 'Time:', time.time()-t0)

    # Test
    model.train(False)
    t = []
    p = []
    for batch, (sample, ref) in enumerate(test_loader):
        sample, ref = sample.to('cuda').unsqueeze(1), ref.to('cuda')
        ref = ref.to('cuda') 
        pred = model(sample)
        pred = F.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
        ref = ref.argmax(dim=1)
        t.append(ref.flatten().unsqueeze(0))
        p.append(pred.flatten().unsqueeze(0))
    t = cat(t, dim=1).squeeze()
    p = cat(p, dim=1).squeeze()
    print(classification_report(t.cpu(), p.cpu()))
    print(confusion_matrix(t.cpu(), p.cpu()))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
