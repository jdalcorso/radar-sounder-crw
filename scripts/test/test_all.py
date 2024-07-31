
import argparse
import time
import torch
import numpy as np
from torch.cuda import device_count, is_available
from torch.nn import DataParallel
from torch import load, manual_seed, cat, logical_and, flip, device, sum
from utils import create_dataset, create_model, get_reference, propagate, plot, cherry_pick
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from sklearn.metrics import classification_report, confusion_matrix

manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 3, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(32,32), type=int)
    parser.add_argument('--seq_length', default=50, type=int)
    parser.add_argument('--overlap', default=(24,0), nargs = '+', type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=100, type=int) # 80-25-0.01-30 works with MiguelDS
    parser.add_argument('-r','--radius', default=13, type=int)
    parser.add_argument('-t','--temp', default=0.1, type=float)
    parser.add_argument('-k','--knn', default=30, type=int)
    # Paths
    parser.add_argument('--model_path', default = '/home/jordydalcorso/workspace/crw/resources/models/sharad32.pt')
    parser.add_argument('--output_folder', default = '/home/jordydalcorso/workspace/crw/resources/output/')
    # Dev
    parser.add_argument('--pos_embed', default = False)
    parser.add_argument('--remove_unc', default = True) # Remove uncertainty class from reports
    parser.add_argument('--flip', default = False) # Flip the full radargram and test on the flipped version
    parser.add_argument('--use_last', default = False) # Use last sample as reference for each rg
    parser.add_argument('--dataset_full',default = True)
    parser.add_argument('--correction', default = True) # Does automatic cpoint detection and correction (skipped if cherry=True)
    # Debug
    parser.add_argument('--boundary_correction', default = True) # Relational features (SHARAD only)
    parser.add_argument('--clamp', default = 55) # Whether to set a limit to the number of bedrock pixels per rangeline
    parser.add_argument('--cherry', default = True) # Whether to cherry pick (user-provide) change points
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
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap, full = args.dataset_full, flip=args.flip)
    dummy = dataset[0].to('cuda') # dummy
    T, N, H, W = dummy.shape
    nclasses, seg = get_reference(id = args.dataset, h = N*H, w = 0, flip=args.flip, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)

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

    tot_rg = seg.shape[-1]//rg_len
    print('Num of radargrams:',tot_rg,'Radargram length:', rg_len)

    seg = seg[:,:tot_rg*rg_len]
    up = transforms.Resize((seg.shape[0],rg_len), interpolation = InterpolationMode.NEAREST)

    # Compute segmentation for each radargram
    seg_list = []
    xent_list = []
    change_list = []
    if args.dataset_full:
        rg_idx_list = range(0,len(dataset),args.seq_length)
    else:
        rg_idx_list = range(tot_rg)
    print('\nList of items picked from the dataset:',list(rg_idx_list),'\n')

    for t in range(len(rg_idx_list)):
        print('Radargram',t)
        seq = dataset[rg_idx_list[t]].to('cuda')
        seg_ref = seg[:rg_h,rg_len * t:rg_len * t + W]
        final_prediction, xent, change_idx = propagate(seq, seg_ref, encoder, lp, nclasses, args.pos_embed, use_last = False)
        final_prediction = up(final_prediction[None]).squeeze()
        plot(img = final_prediction.cpu(), save = args.output_folder+'im'+str(t)+'.png', seg = seg[:,rg_len * t:rg_len * t + rg_len], dataset=args.dataset)
        seg_list.append(final_prediction)
        xent_list.append(xent)
        change_list.append(change_idx)

    if args.cherry:
        change_list = cherry_pick(change_list, args.dataset, args.patch_size[1])

    # Correction step
    if args.correction:
        print('\nCorrection step')
        print('Change point for each radargram:',change_list)
        change_list = [x for x in change_list]

        for t, change_idx in enumerate(change_list):
            print('Radargram',t)
            if change_idx is not None:
                small_length = args.seq_length - change_idx
                pixel_offset = small_length*(args.patch_size[-1]-args.overlap[-1])
                try:
                    seq = dataset.get_smaller_item(rg_idx_list[t],small_length).to('cuda')
                    seg_ref = seg[:,rg_len * t + rg_len-pixel_offset:rg_len * t + rg_len-pixel_offset + W]
                    corrected_prediction, _, _ = propagate(seq, seg_ref, encoder, lp, nclasses, args.pos_embed, use_last = False)
                    seg_list[t][:,rg_len-pixel_offset:] = resize(corrected_prediction[None],
                                                                size = (seg.shape[0],pixel_offset),
                                                                interpolation = InterpolationMode.NEAREST).squeeze()
                    plot(img = seg_list[t].cpu(), save = args.output_folder+'im'+str(t)+'c.png', seg = seg[:,rg_len * t:rg_len * t + rg_len], dataset=args.dataset)
                except:
                    pass

    # Concat seg_list to match the dimension of the full ground truth segmentation
    final_pred = cat(seg_list, dim = 1)

    # Correct boundaries on the basis of relational features
    if args.boundary_correction and args.dataset == 3:
        for i in range(final_pred.shape[1]):
            try:
                first_return_index = (final_pred[:,i] == 1).nonzero(as_tuple=True)[0][0]
                final_pred[:first_return_index,i] = 0
            except:
                pass
            try:    
                last_return_index = (final_pred[:,i] == 2).nonzero(as_tuple=True)[0][-1]
                final_pred[last_return_index:,i] = 3
            except:
                pass
            try:
                last_strong_index = (final_pred[:,i] == 1).nonzero(as_tuple=True)[0][-1]
                first_bedrock_index = (final_pred[:,i] == 2).nonzero(as_tuple=True)[0][0]
                final_pred[last_strong_index:first_bedrock_index,i] = 4
            except:
                pass
        print('\nCorrected boundaries!\n')

    # Save the map and flatten
    torch.save(final_pred.to(torch.int8), args.output_folder+'predicted_map.pt')
    final_pred = final_pred.flatten()
    gt_seg = seg.flatten()

    if args.use_last:
        print('Reversed step\n')
        # Compute segmentation for each reversed radargram
        seg = seg.unfold(dimension = 1, size = rg_len, step = rg_len)
        seg = flip(seg, (-1,)).view(seg.shape[0],-1)
        seg_list = []
        for t in range(len(rg_idx_list)):
            print('Radargram',t)
            seq = dataset[rg_idx_list[t]].to('cuda')
            seg_ref = seg[:,rg_len * t:rg_len * t + W]
            final_prediction, _, _ = propagate(seq, seg_ref, encoder, lp, nclasses, args.pos_embed, use_last = True)
            final_prediction = up(final_prediction[None]).squeeze()
            plot(img = final_prediction.cpu(), save = args.output_folder+'im'+str(t)+'r.png', seg = seg[:,rg_len * t:rg_len * t + rg_len], dataset=args.dataset)
            seg_list.append(final_prediction)

        pred_seg_rev = cat(seg_list, dim = 1).unfold(dimension = 1, size = rg_len, step = rg_len)
        pred_seg_rev = flip(pred_seg_rev, (-1,)).view(pred_seg_rev.shape[0],-1).cpu()
        # Mask
        if args.dataset == 0:
            mask = pred_seg_rev.flatten() == 2
        if args.dataset == 1:
            mask = torch.logical_and(pred_seg_rev.flatten() == 2,final_pred.cpu()!=3) # Mask=1 at bedrock pixels
            mask2 = torch.all(pred_seg_rev != 4, axis=0).unsqueeze(0).repeat([pred_seg_rev.shape[0],1]).flatten() # Mask bedrock pixel under floating ice
            mask = torch.logical_and(mask, mask2)
        if args.dataset == 3:
            mask = pred_seg_rev.flatten() == 2
            mask[:len(mask)//2] = 0
        final_pred[mask] = 2
    
    # Clamp bedrock
    if args.clamp > 0 and args.dataset == 1:
        final_pred = final_pred.view(1536, 103680).cpu()
        clamp = args.clamp # clamp
        for i in range(final_pred.shape[1]):
            if i%1000==0:
                print('Column',i)
            idx = np.where(final_pred[:,i] == 2)[0]
            final_pred[:,i][idx[clamp:]] = 1
        final_pred = final_pred.flatten()
        
    # Remove uncertain class (CRESIS radargrams only)
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
        if args.dataset == 3:
            gt = gt_seg
            pred = final_pred
    else:
        gt = gt_seg
        pred = final_pred

    # Compute reports
    print('Time elapsed (inference only):',time.time()-tim)
    print('Computing reports ...')
    print('')
    print(classification_report(gt, pred.cpu()))
    print(confusion_matrix(gt, pred.cpu()))
    print('\nTime elapsed (inference + metrics):',time.time()-tim)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
