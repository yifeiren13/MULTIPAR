import torch
from torch import nn


class NonnegativeSigmoid(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x):
        return 2 / (1 + torch.exp(-self.gamma * x)) - 1


class MultiPARAFAC2(nn.Module):
    def __init__(self,
                 num_visits,
                 num_feats,
                 rank,
                 alpha,
                 gamma,
                 is_projector=False):
        super().__init__()
        self.num_pts = len(num_visits)
        self.num_visits = num_visits
        self.num_feats = num_feats
        self.rank = rank
        self.gamma = gamma
        self.alpha = alpha

        self.U = nn.Parameter(torch.rand(self.num_pts, max(num_visits), rank))
        self.S = nn.Parameter(torch.rand(self.num_pts, rank) * 30 / rank)
        self.V = nn.Parameter(torch.rand(num_feats, rank))
        #self.U = nn.Parameter(Uini)
        #self.S = torch.nn.Parameter(Wini)
        #self.V = nn.Parameter(Vini)
        self.Phi = nn.Parameter(torch.rand(rank, rank), requires_grad=False)

        self.sigmoid = NonnegativeSigmoid(gamma)

        for i, num_visit in enumerate(num_visits):
            self.U.data[i, num_visit:] = 0
        
        if not is_projector:
            self.update_phi()

    def forward(self, pids):
        out = torch.einsum('ptr,pr,fr->ptf', self.U[pids], self.S[pids], self.V)
        #out = self.sigmoid(out)
        return out

    def projection(self):
        self.U.data = self.U.data.clamp(min=0, max=self.alpha)
        self.S.data = self.S.data.clamp(min=0, max=self.alpha)
        self.V.data = self.V.data.clamp(min=0, max=self.alpha)


    def update_phi(self):
        if self.rank <= 200:  # use GPU with small ranks
            self.Phi.data = (torch.transpose(self.U.data, 1, 2) @ self.U.data).mean(dim=0)
        else:  # use CPU to avoid insufficient VRAM error
            Phi = (torch.transpose(self.U.data.cpu(), 1, 2) @ self.U.data.cpu()).mean(dim=0)
            self.Phi.data = Phi.to(self.Phi.data.device)
    
    def uniqueness_regularization(self, pids):
        U = self.U[pids]
        reg = torch.norm(torch.transpose(U, 1, 2) @ U - self.Phi.unsqueeze(0)) ** 2
        return reg / pids.shape[0]



    
    
    
class SmoothnessConstraint(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, X, seq_len, deltas, norm_p=1):
        L = torch.zeros(X.shape[1], X.shape[1]+1)
        L[:, :-1] = torch.eye(X.shape[1])
        L[:, 1:] += -1 * torch.eye(X.shape[1])
        L = L.unsqueeze(0).repeat(seq_len.shape[0], 1, 1)
        L[torch.arange(seq_len.shape[0]), (seq_len-1).long()] = 0
        L = L[:, :-1, :-1]
        L = L.to(X.device)
        smoothness_mat = torch.exp(-self.beta * deltas[:, 1:].unsqueeze(2)) * (L @ X)
        smoothness = (smoothness_mat).norm(p=norm_p, dim=1) ** norm_p
        return smoothness.sum()
        #return X





class TemporalDependency(nn.Module):
    def __init__(self, rank, nlayers, nhidden, dropout):
        super(TemporalDependency, self).__init__()

        self.nlayers = nlayers
        self.nhid = nhidden

        self.rnn = nn.GRU(input_size=rank,
                           hidden_size=nhidden,
                           num_layers=nlayers,
                           dropout=dropout,
                           batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(nhidden, rank),
            nn.ReLU()
        )

        # self.decoder = nn.Linear(nhidden, rank)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, Ws):
        train_loss = 0.0
        for Wp in Ws:
            inputs, targets = Wp[:-1, :], Wp[1:, :]  # seq_len x n_dim
            seq_len, n_dims = inputs.size()

            hidden = self.init_hidden(1)
            # seq_len x n_dims --> 1 x seq_len x n_dims
            outputs, _ = self.rnn(inputs.unsqueeze(0), hidden)
            logits = self.decoder(outputs.contiguous().view(-1, self.nhid))
            loss = self.loss(logits, targets)
            train_loss += loss
        return train_loss

    def init_hidden(self, batch_sz):
        size = (self.nlayers, batch_sz, self.nhid)
        weight = next(self.parameters())
        return (weight.new_zeros(*size))
        #return (weight.new_zeros(*size),
                #weight.new_zeros(*size))


    def loss(self, input, target):
        return torch.mean((input - target) ** 2)

