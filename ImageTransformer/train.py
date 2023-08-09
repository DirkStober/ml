import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from model import *

torch.manual_seed(4032)

ds = DataSet()

model_file = "image_transformer.pt"


if((os.path.isfile(model_file))):
    m = torch.load(model_file)
else:
    m = IT1D()





# -------------------- Training Parameters ---------------- #
lr = 2e-4
max_iters = 10000

optimizer = torch.optim.AdamW(m.parameters(),lr=lr)


for steps in range(max_iters):
    xb,yb,pos = ds.get_batch('train')
    logits, loss = m(xb,pos,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if(steps % 1000 == 0):
        print(f"Step: {steps}/{max_iters} Loss : {ds.estimate_loss(m)}")
    if(steps == max_iters - 1):
        print(f"Step: {steps}/{max_iters} Loss : {ds.estimate_loss(m)}")

# Save the model
torch.save(m,model_file)

imgs = [m.generate() for _ in range(9)]
fig,axs = plt.subplots(3,3)
[axs[i//3][i%3].imshow(imgs[i],cmap='gray') for i in range(9)]
plt.show()

