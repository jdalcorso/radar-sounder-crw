# Test and plot qualitative results of radargrams of SHARAD dataset

import argparse
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import device_count, is_available
from torch.nn import DataParallel
from torch import load, manual_seed, logical_and, flip, device
from utils import create_model, propagate, plot
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test SHARAD', add_help=False)
    parser.add_argument('--patch_size', default=(16,16), type=int)
    parser.add_argument('--seq_length', default=100, type=int) # 100 * 16 = 1600
    parser.add_argument('--overlap', default=(8,0), nargs = '+', type=int)
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=100, type=int) # 80-25-0.01-30 works with MiguelDS
    parser.add_argument('-r','--radius', default=10, type=int)
    parser.add_argument('-t','--temp', default=0.1, type=float)
    parser.add_argument('-k','--knn', default=20, type=int)
    # Dev
    parser.add_argument('--use_last', default = True) # Use last sample as reference for each rg
    # Folder
    parser.add_argument('--input_folder', default = '/home/jordydalcorso/workspace/crw/resources/input/')
    parser.add_argument('--output_folder', default = '/home/jordydalcorso/workspace/crw/resources/output/')
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
    encoder.load_state_dict(load('/home/jordydalcorso/workspace/crw/resources/models/sharad16_3.pt'))

    # Dataset and reference
    nclasses = 5
    H, W = args.patch_size
    OH, OW = args.overlap
    # Radargram
    rg1 = torch.flip(load(args.input_folder + 's_1.pt').float().to(dev),dims=(1,))
    rg2 = load(args.input_folder + 's_4.pt').float().to(dev)
    rg3 = load(args.input_folder + 's_3.pt').float().to(dev)
    # Forward reference
    sg1 = torch.flip(load(args.input_folder + 's_1ref.pt').to(dev),dims=(1,))
    sg2 = load(args.input_folder + 's_4ref.pt').to(dev)
    sg3 = load(args.input_folder + 's_3ref.pt').to(dev)

    sg = (sg1,sg2,sg3)
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
        plt.imshow(-xent, cmap = 'gray')
        plt.gca().set_aspect(xent.shape[1]/xent.shape[0]*0.77)
        plt.colorbar()
        plt.savefig(args.output_folder+'sharad_xent'+str(t)+'.png')
        plt.close()
        seg_list.append(final_prediction)
        xent_list.append(xent)
        change_list.append(change_idx)

    print('Predicted change list:', change_list)
    change_list[0] = 80
    change_list[1] = 67
    change_list[2] = 98
    
    # Correction step
    print('Correction step')
    print(change_list)
    change_list = [x for x in change_list]

    for t, change_idx in enumerate(change_list):
        print('Radargram',t)
        if change_idx is not None:
            small_length = args.seq_length - change_idx
            pixel_offset = small_length*(args.patch_size[-1]-args.overlap[-1])
            seq = rg[t][change_idx:]
            seg_ref = sg[t][:,rg_len-pixel_offset:rg_len-pixel_offset + W]
            corrected_prediction, _, _ = propagate(seq, seg_ref, encoder, lp, nclasses, do_pos_embed=False, use_last = False)
            seg_list[t][:,rg_len-pixel_offset:] = resize(corrected_prediction[None],
                                                        size = (rg_h,pixel_offset),
                                                        interpolation = InterpolationMode.NEAREST).squeeze()
            plot(img = seg_list[t].cpu(), save = args.output_folder+'sharad_res'+str(t)+'.png', dataset=3)

    torch.save(seg_list, args.output_folder+'s_res.pt')
    torch.save(xent_list, args.output_folder+'s_xent.pt')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)