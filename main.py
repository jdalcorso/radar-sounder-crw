from torch.nn.functional import normalize, softmax, cross_entropy
from torch import einsum, cat, flip, eye, bmm, manual_seed, permute, rand, zeros
from torch import tensor, clone, save, load, matmul, max, argmax
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CNN, Resnet
from dataset import MCORDS1Dataset
from imported.labelprop import LabelPropVOS_CRW
from imported.crw import CRW
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

mod = Resnet()
mod.predict()

manual_seed(11)
model = CNN()
model = model.to('cuda')

dataset = MCORDS1Dataset(length = 40, dim = (12,12), overlap = (0,0))
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)

optimizer = Adam(model.parameters(), lr=0.001)

# sample = dataset[0]
# print(sample.shape) # 10x33x24x24
# plt.imshow(sample[0,0,:,:])
# plt.savefig('sample.png')
# plt.close()

todo = 'test'

if todo=='train':
    epochs = 20
    tau = 0.01
    model.train(True)
    loss_tot = []
    for epoch in range(epochs):
        loss_epoch = []
        for batch, seq in enumerate(dataloader):
            seq = seq.to('cuda')
            B, T, N, H, W = seq.shape    
            emb = model(seq.view(-1, H, W).unsqueeze(1)).view(B, T, N, -1)  # B x T x N x C                 # TODO QUESTO FORSE E' SBAGLIATO, C NON SI SPOSTA AUTOMATICAMENTE IN QUELLA POSIZIONE
            emb = normalize(emb, dim = -1) # L2 normalisation: now emb has L2norm=1 on C dimension
            emb = permute(emb, [0, 3, 1, 2])                                # B x C x T x N

            # Transition from t to t+1. We do a matrix product on the C dimension (i.e. computing cosine similarities)
            A = einsum('bctn,bctm->btnm', emb[:,:,:-1], emb[:,:,1:])/tau     # B x T-1 x N x N
            # Transition energies for palindrome graphs. Sum of rows is STILL not 1. We dont have probabilities yet, we have cosine similarities
            AA = cat((A, flip(A,dims = [1]).transpose(-1,-2)), dim = 1)   # B x 2*T-2 x N x N
            #AA[rand([B, 2*T-2, N, N])< 0.1] = -1e10    # Edge Dropout
            loss = 0

            # For each of the k palindrome paths
            for k in range(1):
                At = zeros(1,N,N, device = 'cuda')
                At[0,:,:] = eye(N)
                At = At.repeat([B,1,1]) # now At is B identity matrices stacked
                if k == 0:
                    I = clone(At)
                # Do walk
                for t in range(k, 2*T-2-k):
                    At = bmm(softmax(AA[:,t], dim = -1), At)
                loss += cross_entropy(input = At, target = I)

            loss_epoch.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch = tensor(loss_epoch).mean()
        loss_tot.append(loss_epoch)

        print('Epoch:',epoch,'Loss:',loss_epoch.item())
    save(model.state_dict(), './crw/latest.pt')

if todo == 'test':
    # Define model
    nclasses = 4
    model = CNN()
    model.to('cuda')
    model.load_state_dict(load('./crw/latest.pt'))
    model.train(False)
    seq = dataset[0].to('cuda')
    T, N, H, W = seq.shape
    emb = model(seq.view(-1, H, W).unsqueeze(1)).view(T,N,-1)
    emb = normalize(emb, dim = -1) # L2
    seg = load('/data/MCoRDS1_2010_DC8/SG2_MCoRDS1_2010_DC8.pt')[:N*H,:T*W]
    mask = zeros(nclasses, N*H, T*W, device = 'cuda')
    for class_idx in range(0, nclasses):
        m = (seg == class_idx).unsqueeze(0).float()
        mask[class_idx, :, :] = m
    mask = mask.unsqueeze(0)

    # Define label propagation method TODO: Implement context by slicing feats/masks
    cfg = {
        'CXT_SIZE' : 10, 
        'RADIUS' : 4,
        'TEMP' : 0.01,
        'KNN' : 10,
    }
    lp = LabelPropVOS_CRW(cfg)

    feats = []
    masks = []

    final_prediction = zeros(N,T, device = 'cuda')
    down = transforms.Resize((N,1), interpolation = InterpolationMode.NEAREST)

    # add reference mask and features
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

        # add new mask (no more one-hot) to the final prediction
        final_prediction[:,t] = argmax(mask, dim = 1).squeeze()

    plt.subplot(121)
    plt.imshow(final_prediction.cpu())
    # plt.imshow(seg)
    plt.subplot(122)
    img = zeros((N*H,T*W))
    for t in range(T):
        for n in range(N):
            img[n*H:n*H+H,t*W:t*W+W] = seq[t,n,:,:]
    plt.imshow(img)
    plt.savefig('reco.png')
    plt.close()





print('Done')

