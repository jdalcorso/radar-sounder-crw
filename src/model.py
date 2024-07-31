import torch.nn as nn
from torch import einsum, cat, flip, eye, bmm, permute, zeros
from torch.nn.functional import normalize, softmax, cross_entropy
from utils import pos_embed

class CRW(nn.Module):
    def __init__(self, encoder, tau, pos_embed, only_a = False):
        super(CRW, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.pos_embed = pos_embed
        self.only_a = only_a
        

    def forward(self, seq):
        B, T, N, H, W = seq.shape   
        seq = seq.reshape(-1, H, W).unsqueeze(1) # BT x 1 x H x W
        if self.pos_embed:
            seq = pos_embed(seq)
        emb = self.encoder(seq)  # B x T x N x C
        emb = emb.reshape(B, T, N, -1)
        emb = normalize(emb, dim = -1) # L2 normalisation: now emb has L2norm=1 on C dimension
        emb = permute(emb, [0, 3, 1, 2]) # B x C x T x N

        # Transition from t to t+1. We do a matrix product on the C dimension (i.e. computing cosine similarities)
        A = einsum('bctn,bctm->btnm', emb[:,:,:-1], emb[:,:,1:])/self.tau     # B x T-1 x N x N
        if self.only_a:
            return A
        
        # Transition energies for palindrome graphs. Sum of rows is STILL not 1. We dont have probabilities yet, we have cosine similarities
        AA = cat((A, flip(A,dims = [1]).transpose(-1,-2)), dim = 1)   # B x 2*T-2 x N x N
        # AA[rand([B, 2*T-2, N, N])< 0.2] = -1e10    # Edge Dropout (worsen performance)
        loss = 0

        for k in range(1,T-1):
            At = zeros(1,N,N, device = 'cuda')
            At[0,:,:] = eye(N)
            At = At.repeat([B,1,1]) # now At is B identity matrices stacked
            I = At # softmax(At/0.1,dim=1)
            #Do walk
            AA_this = cat([AA[:,:k],AA[:,-k:]], dim=1)
            for t in range(1,2*k):
                current = AA_this[:,t]
                At = bmm(softmax(current, dim = -1), At)
            loss += cross_entropy(input = At, target = I)
        return loss/N, A