from torch.cuda import device_count
from torch.nn import DataParallel
from torch import zeros, load,  manual_seed
from utils import create_dataset, create_model, get_reference, propagate, plot
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import argparse
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 0, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(48,16), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(47,8), type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=80, type=int) # 10 - 4 - 0.01 - 10 works with CNN()
    parser.add_argument('-r','--radius', default=200, type=int)
    parser.add_argument('-t','--temp', default=0.01, type=int)
    parser.add_argument('-k','--knn', default=5, type=int)
    # Paths
    parser.add_argument('--model_path', default = '/home/jordydalcorso/workspace/crw/latest.pt')
    # Dev
    parser.add_argument('--pos_embed', default = False)
    return parser

def main(args):
    # Model 
    encoder = create_model(args.model, args.pos_embed)
    encoder.to('cuda')
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    encoder.load_state_dict(load(args.model_path))
    encoder.train(False)

    # Dataset
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    seq = dataset[0].to('cuda')
    T, N, H, W = seq.shape
    rg_len = T * (W - args.overlap[-1]) + args.overlap[-1]
    rg_h   = N * (H - args.overlap[0]) + args.overlap[0]

    # Obtain image (to plot)
    img = zeros((rg_h,rg_len))
    for t in range(T):
        for n in range(N):
            img[n*(H-args.overlap[0]) : n*(H-args.overlap[0])+H, t*(W - args.overlap[-1]) : t*(W - args.overlap[-1])+W] = seq[t,n,:,:]

    nclasses, seg = get_reference(id = args.dataset, h = rg_h, w = rg_len)
    cfg = {
        'CXT_SIZE' : args.cxt_size, 
        'RADIUS' : args.radius,
        'TEMP' : args.temp,
        'KNN' : args.knn,
    }
    lp = LabelPropVOS_CRW(cfg)

    up = transforms.Resize((seg.shape[0],rg_len), interpolation = InterpolationMode.NEAREST)
    seg_ref = seg[:rg_h,:W]
    final_prediction, _, _ = propagate(seq, seg_ref, encoder, lp, nclasses, args.pos_embed, use_last = False)
    final_prediction = up(final_prediction[None]).squeeze()
    plot(img = final_prediction.cpu(), save = '/home/jordydalcorso/workspace/crw/output/_reco.png', seg = seg, dataset=args.dataset)
    print('Test done.')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
