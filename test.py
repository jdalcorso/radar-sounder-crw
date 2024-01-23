from torch.nn.functional import normalize
from torch import permute, zeros, load, argmax, manual_seed
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import CNN, Resnet
from dataset import MCORDS1Dataset
from imported.labelprop import LabelPropVOS_CRW
from imported.crw import CRW
import argparse
import matplotlib.pyplot as plt
manual_seed(11)

def get_args_parser():
    parser = argparse.ArgumentParser('CRW Test', add_help=False)
    # Data
    parser.add_argument('--patch_size', default=(12,12), type=int)
    parser.add_argument('--seq_length', default=80, type=int)
    parser.add_argument('--overlap', default=(0,0), type=int) # Should not be changed
    # Label propagation cfg
    parser.add_argument('-c','--cxt_size', default=10, type=int)
    parser.add_argument('-r','--radius', default=4, type=int)
    parser.add_argument('-t','--temp', default=0.01, type=int)
    parser.add_argument('-k','--knn', default=10, type=int)
    # Paths
    parser.add_argument('--model_path', default = './crw/latest.pt')
    parser.add_argument('--seg_path', default = '/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')
    return parser

def main(args):
    # Model 
    nclasses = 4
    model = CNN()
    model.to('cuda')
    model.load_state_dict(load(args.model_path))
    model.train(False)

    # Dataset
    dataset = MCORDS1Dataset(length = args.seq_length, dim = args.patch_size, overlap = args.overlap)
    seq = dataset[0].to('cuda')

    # Obtain embeddings and reference mask
    T, N, H, W = seq.shape
    emb = model(seq.view(-1, H, W).unsqueeze(1)).view(T,N,-1)
    emb = normalize(emb, dim = -1) # L2
    seg = load(args.seg_path)[:N*H,:T*W]
    mask = zeros(nclasses, N*H, T*W, device = 'cuda')
    for class_idx in range(0, nclasses):
        m = (seg == class_idx).unsqueeze(0).float()
        mask[class_idx, :, :] = m
    mask = mask.unsqueeze(0)

    # Define label propagation method TODO: Implement context by slicing feats/masks
    cfg = {
        'CXT_SIZE' : args.cxt_size, 
        'RADIUS' : args.radius,
        'TEMP' : args.temp,
        'KNN' : args.knn,
    }
    lp = LabelPropVOS_CRW(cfg)

    feats = []
    masks = []

    final_prediction = zeros(N,T, device = 'cuda')
    down = transforms.Resize((N,1), interpolation = InterpolationMode.NEAREST)

    # Add reference mask and features
    label = down(seg[:,0*W:0*W+W].unsqueeze(0)).squeeze(0).to('cuda')
    final_prediction[:,0] = label.squeeze(1)
    mask = zeros(nclasses, N, 1, device = 'cuda')
    for class_idx in range(0, nclasses):
        m = (label == class_idx).unsqueeze(0).float()
        mask[class_idx, :, :] = m
    mask = mask.unsqueeze(0)
    feat = permute(emb[0,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
    feats.append(feat)
    masks.append(mask)

    for t in range(1,T):
        print('Range-line:',t)    
        feat = permute(emb[t,:].unsqueeze(0).unsqueeze(0),[0, 3, 2, 1])
        mask = lp.predict(feats = feats, masks = masks, curr_feat = feat)

        feats.append(feat)
        masks.append(mask)

        # Add new mask (no more one-hot) to the final prediction
        final_prediction[:,t] = argmax(mask, dim = 1).squeeze()

    plt.figure(figsize = (13,13))
    plt.subplot(211)
    plt.imshow(final_prediction.cpu())
    # plt.imshow(seg)
    plt.subplot(212)
    img = zeros((N*H,T*W))
    for t in range(T):
        for n in range(N):
            img[n*H:n*H+H,t*W:t*W+W] = seq[t,n,:,:]
    plt.imshow(img)
    plt.savefig('./crw/reco.png')
    plt.close()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
