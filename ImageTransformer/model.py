import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

#-------------- Hyperparameters -------------- #
block_size = 64
batch_size = 64
img_size = 27
n_embd = 64
n_head = 8
n_blocks = 4
eval_iters = 100
dropout = 0.2


# Num_embd//num_head > 0 and should be an integer
# num_head = 1
# output_len = 256
#-----------------------------------------------#
def convertDS(ds,classid):
    g =[[] for _ in range(10)]
    {g[j].append(i) for i,j in enumerate(ds.targets)}
    in_data = ds.data[g[classid]]
    return in_data

class DataSet():
    def __init__(self):
        mnist_train = torchvision.datasets.MNIST(root='../data',train=True,
                download = True)
        mnist_test = torchvision.datasets.MNIST(root='../data',train=False,
                download = True)
        self.train_data =   convertDS(mnist_train,0)
        self.val_data =     convertDS(mnist_test,0)

    def get_batch(self,split):
        data = self.train_data if split == 'train' else self.val_data
        i_all= torch.randint(0,data.shape[0],(batch_size,))
        j_img= torch.randint(0,img_size*img_size - block_size -1 ,(batch_size,))
        x = torch.stack([data[i].view(-1)[j:j+block_size] for i,j in zip(i_all,j_img)]).long()
        y = torch.stack([data[i].view(-1)[j + 1:j+block_size + 1]for i,j in zip(i_all,j_img)]).long()
        pos_in = j_img.unsqueeze(-1) + torch.arange(block_size)
        return x,y,pos_in

    @torch.no_grad()
    def estimate_loss(self,model):
        out = {}
        model.eval()
        for split in ['train','val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y,P = self.get_batch(split)
                logits,loss = model(X,P,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

#-----------------------------------------------#
# """ Feed Forward """
# class FeedForward(nn.Module):
#
""" Multi Head Self Attention """
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size  ,bias = False)
        self.query= nn.Linear(n_embd,head_size ,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        # (B,T,C) and tril is size TxT (Block_size,Block_size)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # C**-0.5 --> Scaled attention
        wei = q@k.transpose(-2,-1)*(C**-0.5) # (B,T,C) @ (B,C,T) --> (B,T,T)
        # Stop communicating with the future
        wei = wei.masked_fill(self.tril[:T,:T] == 0,-float('inf'))
        wei = F.softmax(wei,dim=-1)
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd,4*n_embd),
                nn.ReLU(),
                nn.Linear(4*n_embd,n_embd),
                nn.Dropout(dropout),
        )
        
    def forward(self,x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        out = self.sa(self.ln1(x)) + x
        out = self.ffwd(self.ln2(out)) + out
        return  out








""" 1D-Context with positional encoding """
class IT1D(nn.Module):
    # Fill in with Layers
    def __init__(self):
        super().__init__()
        self.pos_embedding_x= nn.Embedding(img_size,n_embd//2)
        self.pos_embedding_y= nn.Embedding(img_size,n_embd//2)
        # self.pos_buffer = nn.Embedding(block_size,n_embd)
        self.value_embedding = nn.Embedding(256,n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.OutputLayer = nn.Linear(n_embd,256)

    def forward(self,idx,pos_in,targets = None):
        B,T = idx.shape # Batch and Tokens
        tok_embd = self.value_embedding(idx)
        xemb = self.pos_embedding_x(pos_in //img_size)
        yemb = self.pos_embedding_y(pos_in % img_size)
        pos_embd = torch.cat([xemb,yemb],dim=-1)

        # pos_embd = self.pos_buffer(torch.arange(T))
        x = tok_embd + pos_embd #(B,T,1) + (1,T,1)
        x = self.blocks(x)
        logits = self.OutputLayer(x)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx = torch.zeros(1,1,dtype=torch.long)):
        for i in range(img_size*img_size):
            # Make sure to only input the last 4 values
            idx_in = idx[:,-block_size:]
            logits,_ = self(idx_in,torch.tensor(i))
            logits= logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        # Extract the image
        idx.squeeze()
        img = idx[:,1:].view(img_size,img_size)
        return img
