
from torch.cuda import device_count, is_available
from torch.nn import DataParallel
from torch import load, manual_seed, cat, logical_and, flip, device
from utils import create_dataset, create_model, get_reference, propagate, plot
from model import CRW
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import time
manual_seed(11)


def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 0, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(31,0), type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=80, type=int) # 80-25-0.01-30 works with MiguelDS
    parser.add_argument('-r','--radius', default=32, type=int)
    parser.add_argument('-t','--temp', default=0.01, type=float)
    parser.add_argument('-k','--knn', default=10, type=int)
    # Paths
    parser.add_argument('--model_path', default = './crw/latest.pt')
    # Dev
    parser.add_argument('--pos_embed', default = True)
    parser.add_argument('--remove_unc', default = True) # Remove uncertainty class from reports
    parser.add_argument('--flip', default = False) # Flip the full radargram and test on the flipped version
    parser.add_argument('--use_last', default = True) # Use last sample as reference for each rg
    return parser


def main(args):
    tim = time.time()
    print(args)

    # Model
    dev = device('cuda' if is_available() else 'cpu')
    encoder = create_model(args.model, args.pos_embed)
    encoder.to(dev)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    encoder.load_state_dict(load(args.model_path))

    # Dataset and reference
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap, flip=args.flip)
    dummy = dataset[0].to('cuda') # dummy
    T, N, H, W = dummy.shape
    nclasses, seg = get_reference(id = args.dataset, h = N*H, w = 0, flip=args.flip)

    # Label propagation method
    cfg = {
        'CXT_SIZE' : args.cxt_size, 
        'RADIUS' : args.radius,
        'TEMP' : args.temp,
        'KNN' : args.knn,
    }
    lp = LabelPropVOS_CRW(cfg)

    # Compute the number of total radargrams (with first sample)
    rg_len = ((args.seq_length - 1) * (args.patch_size[-1] - args.overlap[-1]) + args.patch_size[-1]) # in pixels
    tot_rg = seg.shape[-1]//rg_len
    print('Num of radargrams:',tot_rg,'Radargram length:', rg_len)

    seg = seg[:,:tot_rg*rg_len]

    up = transforms.Resize((seg.shape[0],rg_len), interpolation = InterpolationMode.NEAREST)

    # Compute segmentation for each radargram
    seg_list = []
    for t in range(tot_rg):
        print('Radargram',t)
        seq = dataset[t].to('cuda')
        final_prediction = propagate(seq, t, seg, encoder, lp, nclasses, rg_len, args.pos_embed, use_last = False)
        final_prediction = up(final_prediction[None]).squeeze()
        plot(img = final_prediction.cpu(), save = './crw/output/im'+str(t)+'.png', seg = seg[:,rg_len * t:rg_len * t + W*args.seq_length])
        seg_list.append(final_prediction)

    # Concat seg_list to match the dimension of the full ground truth segmentation
    final_pred = cat(seg_list, dim = 1).flatten()
    gt_seg = seg.flatten()

    if args.use_last:
        print('Reversed')
        # Compute segmentation for each reversed radargram
        seg = seg.unfold(dimension = 1, size = rg_len, step = rg_len)
        seg = flip(seg, (-1,)).view(seg.shape[0],-1)
        seg_list = []
        for t in range(tot_rg):
            print('Radargram',t)
            seq = dataset[t].to('cuda')
            final_prediction = propagate(seq, t, seg, encoder, lp, nclasses, rg_len, args.pos_embed, use_last = True)
            final_prediction = up(final_prediction[None]).squeeze()
            plot(img = final_prediction.cpu(), save = './crw/output/im'+str(t)+'.png', seg = seg[:,rg_len * t:rg_len * t + W*args.seq_length])
            seg_list.append(final_prediction)

        pred_seg_rev = cat(seg_list, dim = 1).unfold(dimension = 1, size = rg_len, step = rg_len)
        pred_seg_rev = flip(pred_seg_rev, (-1,)).view(pred_seg_rev.shape[0],-1).flatten()
        # Merge predictions
        mask = pred_seg_rev == 2
        final_pred[mask] = 2


    # Remove class 4 (uncertain)
    if args.remove_unc: 
        if args.dataset == 0:
            _, unc_seg = get_reference(id = 2, h = N*H, w = 0, flip=args.flip)
            unc_seg = unc_seg[:,:tot_rg*rg_len]
            mask = (unc_seg != 4).flatten()
            gt = gt_seg[mask]
            pred = final_pred[mask]
        if args.dataset == 1:
            mask = logical_and(gt_seg.view(seg.shape[0],-1) != 5, final_pred.view(seg.shape[0],-1).cpu() != 5)
            mask = mask.flatten()
            gt = gt_seg[mask]
            pred = final_pred[mask]
    else:
        gt = gt_seg
        pred = final_pred

    print(gt.unique(),pred.unique())
    # Compute reports
    print('Computing reports ...')
    print('')
    print(classification_report(gt, pred.cpu()))
    print(confusion_matrix(gt, pred.cpu()))
    print('Time elapsed:',time.time()-tim)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
