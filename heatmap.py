from model import CRW
from torch.cuda import device_count
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy
from torch import eye, zeros, log, load
from utils import create_model, create_dataset, plot, ndiag_matrix
import argparse
import matplotlib.pyplot as plt
import ruptures as rpt

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Heatmap', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 1, type=int, help='0=MCORDS1,1=Miguel')
    parser.add_argument('--model_path', default = '/home/jordydalcorso/workspace/crw/latest.pt')
    # Data
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(16,0), type=int)
    parser.add_argument('--tau', default = 0.1, type = int)
    parser.add_argument('--pos_embed', default = True, type = bool)
    return parser

def main(args):
    encoder = create_model(args.model, args.pos_embed)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    encoder.load_state_dict(load(args.model_path))
    model = CRW(encoder, args.tau, args.pos_embed, only_a=True)
    model = model.to('cuda')
    model.train(False)
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)

    seq = dataset[0].to('cuda')
    T, N, H, W = seq.shape
    A = model(seq.unsqueeze(0)).squeeze(0)
    I = ndiag_matrix(N, 1).cuda()

    img = zeros((N*H,T*W))
    result = zeros(N,T-1)
    for t in range(T-1):
        for n in range(N):
            img[n*H:n*H+H,t*W:t*W+W] = seq[t,n,:,:]
        At = A[t,:,:]
        result[:,t] = (cross_entropy(input = At, target = I, reduction='none'))
    
    plt.figure(figsize = (13,13))
    plt.subplot(411)
    plt.imshow(img, cmap = 'gray', aspect = .21)

    plt.subplot(412)
    plt.imshow(result.detach().cpu(), cmap = 'gray', aspect = .21, interpolation='nearest')
    plt.clim([0,5])

    # Mean of image
    plt.subplot(413)
    step = 32
    size = 160
    aspect = 59
    roll = img.unfold(dimension = 1, step = step, size = size).detach()
    roll = roll.mean(dim=(0,-1))
    plt.plot(roll,'k')
    plt.gca().set_aspect(aspect)

    # Mean of metric
    plt.subplot(414)
    aspect = 5.28
    rolling = result.unfold(dimension = 1, step = 1, size = 5).detach()
    rolling = rolling.mean(dim=(0,-1))
    plt.plot(rolling,'k')
    plt.gca().set_aspect(aspect)

    # Print change point
    algo = rpt.Pelt(model="rbf").fit(rolling)
    result = algo.predict(pen=10)
    plt.vlines(result[0],0.5,4.0,'r')
    plt.tight_layout()
    plt.savefig('/home/jordydalcorso/workspace/crw/_heatmap.png')
    plt.close()
    
    print('Heatmap done.')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)