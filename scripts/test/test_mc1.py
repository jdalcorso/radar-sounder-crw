# Test and plot qualitative results of radargrams of MCORDS1 dataset

import argparse
import time
import torch
import matplotlib.pyplot as plt
from torch.cuda import device_count, is_available
from torch.nn import DataParallel
from torch import load, manual_seed, logical_and, flip, device
from utils import create_model, propagate, plot
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode

manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test MC1', add_help=False)
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=100, type=int) # 100 * 32 = 3200
    parser.add_argument('--overlap', default=(24,0), nargs = '+', type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=80, type=int) # 80-25-0.01-30 works with MiguelDS
    parser.add_argument('-r','--radius', default=30, type=int)
    parser.add_argument('-t','--temp', default=0.1, type=float)
    parser.add_argument('-k','--knn', default=20, type=int)
    # Dev
    parser.add_argument('--use_last', default = True) # Use last sample as reference for each rg
    # Folder
    parser.add_argument('--input_folder', default = '/home/jordydalcorso/workspace/crw/src/input/')
    parser.add_argument('--output_folder', default = '/home/jordydalcorso/workspace/crw/src/output/')
    return parser


def main(args):
    tim = time.time()

    # Model
    dev = device('cuda' if is_available() else 'cpu')
    encoder = create_model(id = 1, pos_embed = False)
    encoder.to(dev)
    num_devices = device_count()
    if num_devices >= 2:
        encoder = DataParallel(encoder)
    encoder.load_state_dict(load('/home/jordydalcorso/workspace/crw/src/models/latestx.pt'))

    # Dataset and reference
    nclasses = 4
    H, W = args.patch_size
    OH, OW = args.overlap
    # Radargram
    rg1 = load(args.input_folder + 'mc1_1.pt').to(dev)
    rg2 = load(args.input_folder + 'mc1_2.pt').to(dev)
    rg3 = load(args.input_folder + 'mc1_3.pt').to(dev)
    # Forward reference
    sg1 = load(args.input_folder + 'mc1_1ref.pt').to(dev)
    sg2 = load(args.input_folder + 'mc1_2ref.pt').to(dev)
    sg3 = load(args.input_folder + 'mc1_3ref.pt').to(dev)
    # Backward reference
    sg1r = load(args.input_folder + 'mc1_1ref_r.pt').to(dev)
    sg2r = load(args.input_folder + 'mc1_2ref_r.pt').to(dev)
    sg3r = load(args.input_folder + 'mc1_3ref_r.pt').to(dev)
    sg = (sg1,sg2,sg3)
    sgr = (sg1r,sg2r,sg3r)
    rg_raw = [rg1,rg2,rg3]
    rg = []
    # Turn radargrams into (T N H W)
    for item in rg_raw:
        item = item.unfold(dimension = 0, size = H, step= H-OH)
        item = item.unfold(dimension = 1, size = W, step= W-OW)
        item = torch.permute(item, [1,0,2,3])
        rg.append(item)

    T, N, H, W = rg[0].shape # TODO

    # Label propagation method
    cfg = {
        'CXT_SIZE' : args.cxt_size, 
        'RADIUS' : args.radius,
        'TEMP' : args.temp,
        'KNN' : args.knn,
    }
    lp = LabelPropVOS_CRW(cfg)

    # Compute the number of total radargrams (with first sample)
    rg_len = T * (W - args.overlap[-1]) + args.overlap[-1]
    rg_h   = N * (H - args.overlap[0])  + args.overlap[0]

    print('Num of radargrams:', 3,'Radargram length:', rg_len)
    up = transforms.Resize((rg_h,rg_len), interpolation = InterpolationMode.NEAREST)

    # Compute segmentation for each radargram
    seg_list, xent_list, change_list = [], [], []

    for t in range(3):
        print('Radargram',t)
        seq = rg[t]
        seg_ref = sg[t][:rg_h,:W]
        final_prediction, xent, change_idx = propagate(seq, seg_ref, encoder, lp, nclasses, do_pos_embed = False, use_last = False)
        final_prediction = up(final_prediction[None]).squeeze()
        plot(img = final_prediction.cpu(), save = args.output_folder+'im'+str(t)+'.png', dataset=0, aspect = 6)
        plt.imshow(xent, interpolation='nearest', cmap = 'gray')
        plt.gca().set_aspect(xent.shape[1]/xent.shape[0]*0.77)
        plt.savefig(args.output_folder+'im'+str(t)+'xent.png')
        plt.close()
        seg_list.append(final_prediction)
        xent_list.append(xent)
        change_list.append(change_idx)

    if args.use_last:
        print('Reversed step')
        # Compute segmentation for each reversed radargram
        seg_list_r = []
        for t in range(3):
            print('Radargram',t)
            seq = rg[t]
            seg_ref = sgr[t][:rg_h,:W]
            final_prediction, _, _ = propagate(seq, seg_ref, encoder, lp, nclasses, do_pos_embed = False, use_last = True)
            final_prediction = up(final_prediction[None]).squeeze()
            final_prediction = flip(final_prediction,(-1,))
            plot(img = final_prediction.cpu(), save = args.output_folder+'im'+str(t)+'r.png', dataset=0, aspect = 6)
            seg_list_r.append(final_prediction)

        seg_list_final = []
        print('Integration step')
        for t in range(len(seg_list)):
            print('Radargram',t)
            seg_list_final.append(seg_list[t])
            # Mask 1
            mask = seg_list_r[t] == 2
            seg_list_final[t][mask] = 2
            # Mask 2
            mask2 = logical_and(seg_list_r[t] == 1,seg_list[t] !=2)
            seg_list_final[t][mask2] = 1
            plot(img = seg_list_final[t].cpu(), save = args.output_folder+'im'+str(t)+'f.png', dataset=0, aspect = 6)

    torch.save(seg_list_final, args.output_folder+'mc1_res.pt')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)