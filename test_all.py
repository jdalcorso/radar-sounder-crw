
from torch.nn.functional import normalize
from torch.cuda import device_count, is_available
from torch.nn import DataParallel
from torch import permute, zeros, load, argmax, manual_seed, device, cat, inference_mode
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils import create_dataset, create_model, get_reference, pos_embed
from imported.labelprop import LabelPropVOS_CRW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix
import torch
import argparse
import time
import matplotlib.pyplot as plt
manual_seed(11)

from imported.crw import CRW

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Meta
    parser.add_argument('--model', default = 1, type=int, help='0=CNN,1=Resnet18')
    parser.add_argument('--dataset', default = 0, type=int, help='0=MCORDS1,1=Miguel')
    # Data
    parser.add_argument('--patch_size', default=(16,16), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(15,15), type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=10, type=int) # 10 - 4 - 0.01 - 10 works with CNN()
    parser.add_argument('-r','--radius', default=2, type=int)
    parser.add_argument('-t','--temp', default=0.01, type=int)
    parser.add_argument('-k','--knn', default=10, type=int)
    # Paths
    parser.add_argument('--model_path', default = './crw/latest.pt')
    # Dev
    parser.add_argument('--pos_embed', default = False)
    parser.add_argument('--remove_unc', default = True)
    return parser



def main(args):
    tim = time.time()

    # Model
    device = torch.device('cuda' if is_available() else 'cpu')
    model = create_model(args.model, args.pos_embed)
    model.to(device)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model.load_state_dict(load(args.model_path))

    # Dataset and reference
    dataset = create_dataset(id = args.dataset, length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    dummy = dataset[0].to('cuda') # dummy
    T, N, H, W = dummy.shape
    nclasses, seg = get_reference(id = args.dataset, h = N*H, w = 0)

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

    down = transforms.Resize((N,1), interpolation = InterpolationMode.NEAREST)
    up = transforms.Resize((seg.shape[0],rg_len), interpolation = InterpolationMode.NEAREST)

    # Compute segmentation for each radargram
    seg_list = []
    for t in range(tot_rg):
        print('Radargram',t)
        seq = dataset[t].to('cuda')
        
        # Obtain image (to plot)
        img = zeros((N*H,T*W))
        for i in range(T):
            for n in range(N):
                img[n*H:n*H+H,i*W:i*W+W] = seq[i,n,:,:]

        # Obtain embeddings and reference mask
        seq = seq.view(-1, H, W).unsqueeze(1)
        if args.pos_embed:
            seq = pos_embed(seq)
        emb = model(seq).view(T,N,-1)
        emb = normalize(emb, dim = -1) # L2

        feats = []
        masks = []

        final_prediction = zeros(N,T, device = 'cuda')

        # Add reference mask and features
        seg_ref = seg[:,rg_len * t:rg_len * t + W]
        #print(rg_len * t, rg_len * t+W)

        label = down(seg_ref.unsqueeze(0)).squeeze(0).to('cuda')
        final_prediction[:,0] = label.squeeze(1)
        mask = zeros(nclasses, N, 1, device = 'cuda')
        for class_idx in range(0, nclasses):
            m = (label == class_idx).unsqueeze(0).float()
            mask[class_idx, :, :] = m
        mask = mask.unsqueeze(0)
        feat = permute(emb[0,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
        feats.append(feat)
        masks.append(mask)

        for n in range(1,T):
            feat = permute(emb[n,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
            mask = lp.predict(feats = feats, masks = masks, curr_feat = feat)

            feats.append(feat)
            masks.append(mask)

            # Add new mask (no more one-hot) to the final prediction
            final_prediction[:,n] = argmax(mask, dim = 1).squeeze()

        final_prediction = up(final_prediction[None]).squeeze()
        plt.imshow(final_prediction.cpu())
        plt.savefig('./crw/output/im'+str(t)+'.png')
        plt.close()
        seg_list.append(final_prediction)

    # Concat seg_list to match the dimension of the full ground truth segmentation
    predicted_seg = cat(seg_list, dim = 1).flatten()
    gt_seg = seg.flatten()

    # Remove class 4 (uncertain)
    if args.remove_unc and args.dataset == 0:
        _, unc_seg = get_reference(id = 2, h = N*H, w = 0)
        unc_seg = unc_seg[:,:tot_rg*rg_len]
        mask = (unc_seg != 4).flatten()
        gt = gt_seg[mask]
        pred = predicted_seg[mask]
    else:
        gt = gt_seg
        pred = predicted_seg

    # Compute reports
    print('Computing reports ...')
    print('')
    print(classification_report(gt.cpu(), pred.cpu()))
    print(confusion_matrix(gt.cpu(), pred.cpu()))
    print('Time elapsed:',time.time()-tim)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
