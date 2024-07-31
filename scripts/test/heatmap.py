from torch.cuda import device_count
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy, normalize
from torch import zeros, load, permute, einsum
from utils import create_model, ndiag_matrix
import numpy as np
import argparse
import matplotlib.pyplot as plt
import ruptures as rpt

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Heatmap', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 1, type=int, help='0=MCORDS1,1=Miguel,3=SHARAD')
    parser.add_argument('--model_path', default = '/home/jordydalcorso/workspace/crw/resources/models/sharad.pt')
    parser.add_argument('--input_folder', default = '/home/jordydalcorso/workspace/crw/resources/input/')
    parser.add_argument('--output_folder', default = '/home/jordydalcorso/workspace/crw/resources/output/')
    # Data
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=100, type=int)
    parser.add_argument('--overlap', default=(24,0), type=int)
    parser.add_argument('--tau', default = 0.1, type = int)
    parser.add_argument('--pos_embed', default = False, type = bool)
    return parser

def main(args):
    encoder = create_model(args.model, args.pos_embed)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    encoder.load_state_dict(load(args.model_path))
    encoder.to('cuda')

    H,W = args.patch_size
    OH,OW = args.overlap
    if args.dataset == 1:
        rg = load(args.input_folder + 'mc3_1.pt').float().to('cuda')[:1000,:1920]
    if args.dataset == 3:
        rg = load('/home/jordydalcorso/workspace/datasets/SHARAD/sharad_north_rg.pt').float().to('cuda')[:,:1920]
    seq = rg.unfold(dimension = 0, size = H, step= H-OH)
    seq = seq.unfold(dimension = 1, size = W, step= W-OW)
    seq = permute(seq, [1,0,2,3])
    T, N, _, _ = seq.shape
    seq = seq.reshape(-1, H, W).unsqueeze(1)

    emb = encoder(seq).view(T,N,-1)
    emb = normalize(emb, dim = -1) # L2

    I = ndiag_matrix(N, 1).cuda()
    xent = zeros(N,T-1, requires_grad = False)
    A = einsum('tnc,tmc->tnm', emb[:,:,:-1], emb[:,:,1:])/args.tau
    for i in range(T-1):
        At = A[i,:,:]
        xent[:,i] = (cross_entropy(input = At, target = I, reduction='none'))

    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(13, 13)

    fs = 20
    aspect = 0.3
    vx = [[0,500,1000,1500],   [10,30,50],       [10,30,50],           [10,30,50]]
    lx = [[0,500,1000,1500],   [10,30,50],       [10,30,50],           [10,30,50]]
    vy = [[0,500,1000],        [0,15,30],        [.2,.4,.6,.8],        [3.55,3.60,3.65]]
    ly = [[0,10,20],           [0,10,20],        [.2,.4,.6,.8],         [0.2,0.4,0.6]]
    ty = ['Time [μs]', 'Time [μs]', 'Mean', 'Metric']
    tx = ['Trace', 'Column', 'Column', 'Column']

    ax[0].imshow(rg.cpu(), cmap = 'gray')

    ax[1].imshow(xent.detach().cpu(), cmap = 'gray', interpolation='nearest')

    # Mean of image
    step = 32
    size = 1
    roll = rg.unfold(dimension = 1, step = step, size = size).detach()
    roll = roll.mean(dim=(0,-1))
    ax[2].plot(roll.cpu(),'k')
    ax[2].grid()
    ax[2].set_xlim(0,len(roll)-1)


    # Mean of metric
    rolling = xent.unfold(dimension = 1, step = 1, size = 1).detach()
    rolling = rolling.mean(dim=(0,-1))
    rolling = xent.mean(dim=0).detach()
    ax[3].plot(rolling,'k')
    ax[3].grid()
    ax[3].set_xlim(0,len(rolling)-1)
    ax[3].vlines(34,3.53,3.65,'r','--')

    i = 0  
    for a in ax:
        a.set_aspect(np.abs(np.diff(a.get_xlim())/np.diff(a.get_ylim()))*aspect)
        a.set_xticks(vx[i])
        a.set_xticklabels(lx[i])
        a.set_yticks(vy[i])
        a.set_yticklabels(ly[i])
        a.tick_params(axis='both', labelsize=fs)
        a.set_xlabel(tx[i], fontsize=fs)
        a.set_ylabel(ty[i], fontsize=fs)
        i+=1

    # Print change point
    algo = rpt.Pelt(model="rbf").fit(rolling)
    res = algo.predict(pen=2)
    plt.tight_layout()
    plt.savefig(args.output_folder+'_heatmap.pdf', format = 'pdf', dpi = 100, bbox_inches='tight')
    plt.savefig(args.output_folder+'_heatmap.png')
    plt.close()

    print('Heatmap done.')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)