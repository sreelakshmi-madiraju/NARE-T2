import torch
from torch import nn
import torch.nn.functional as F
import math
import time
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#write the code for model
class PositionalEncoding(nn.Module):
     """
     From the original doc - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
     """
     def __init__(self, d_model, vocab_size=10000, dropout=0.2):
         super().__init__()
         self.dropout = nn.Dropout(p=dropout)
         pe = torch.zeros(vocab_size, d_model)
         position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
         div_term = torch.exp(
             torch.arange(0, d_model, 2).float()
             * (-math.log(10000.0) / d_model)
         )
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         pe = pe.unsqueeze(0)
         self.register_buffer("pe", pe)
     def forward(self, x):
         x = x + self.pe[:, : x.size(1), :]
         return self.dropout(x)

class PositionalEmbedding(nn.Module):
    """
    Computes trainable positional embedding "
    """

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        batch_size = x.size(0)
        return self.dropout(x+self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1))


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.sigmoid(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)



class Transformer(nn.Module):
    """
    Transformer encoder module for recommendation.
    """

    def __init__(
        self,
        embeddings,
        nhead=8,
        dim_feedforward=32,
        num_layers=2,
        dropout=0.3,
        activation="relu",
        pos_encoding=1,
    ):

        super().__init__()

        vocab_size, d_model = embeddings.size()
        max_len = 15
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        #self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
             d_model=d_model,
             dropout=dropout,vocab_size=vocab_size)
        if pos_encoding ==1:
            self.pos_encoder = PositionalEmbedding(max_len=max_len, d_model=d_model)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.55)
        self.attention_layer = Attention(d_model, max_len, bias = False)
        self.device= device
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first= True,
            # norm_first= True,
            layer_norm_eps=1e-05,
            
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.b = nn.Linear(d_model, 2*d_model, bias=False)
        self.emb_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        mask = torch.where(x > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        
        x1 = self.emb(x) * math.sqrt(self.d_model)
        x1 = self.pos_encoder(x1)
        x1 = self.transformer_encoder(x1)
        
        c_global = self.attention_layer(x1, mask=mask)
        c_local= x1[:,-1,:]
        c_t = torch.cat([c_local,c_global], 1)
        c_t = self.dropout(c_t)
        
        item_embs = self.emb(torch.arange(self.vocab_size).to(self.device))
        scores = torch.matmul(c_t,self.b(item_embs).permute(1, 0))
        return scores

def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()


def evaluate(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=50):
    sum_epoch_loss = 0
    model.train(True)
    start = time.time()
    for i, (seq, target) in enumerate(train_loader):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target in valid_loader:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = evaluate(logits, target, k = 20)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

