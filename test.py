from torch.cuda import device_count
from torch.nn import DataParallel
from torch import zeros, load,  manual_seed
from utils import create_dataset, create_model, get_reference, propagate, plot
from imported.labelprop import LabelPropVOS_CRW
import argparse
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 1, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(16,16), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(0,0), type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=80, type=int) # 10 - 4 - 0.01 - 10 works with CNN()
    parser.add_argument('-r','--radius', default=25, type=int)
    parser.add_argument('-t','--temp', default=0.01, type=int)
    parser.add_argument('-k','--knn', default=30, type=int)
    # Paths
    parser.add_argument('--model_path', default = './crw/latest.pt')
    # Dev
    parser.add_argument('--pos_embed', default = True)
    return parser

def main(args):
    # Model 
    model = create_model(args.model, args.pos_embed)
    model.to('cuda')
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model.load_state_dict(load(args.model_path))
    model.train(False)

    # Dataset
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    seq = dataset[0].to('cuda')
    T, N, H, W = seq.shape
    rg_len = ((args.seq_length - 1) * (args.patch_size[-1] - args.overlap[-1]) + args.patch_size[-1])

    # Obtain image (to plot)
    img = zeros((N*H,T*W))
    for t in range(T):
        for n in range(N):
            img[n*H:n*H+H,t*W:t*W+W] = seq[t,n,:,:]

    nclasses, seg = get_reference(id = args.dataset, h = N*H, w = T*W)
    cfg = {
        'CXT_SIZE' : args.cxt_size, 
        'RADIUS' : args.radius,
        'TEMP' : args.temp,
        'KNN' : args.knn,
    }
    lp = LabelPropVOS_CRW(cfg)

    t = 0
    final_prediction = propagate(seq, t, seg, model, lp, nclasses, rg_len, args.pos_embed, use_last = False)
    plot(img = final_prediction.cpu(), save = './crw/output/a_reco.png', seg = img)
    print('Test done.')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
