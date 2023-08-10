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



pth = "itnet.pt"

# -------------------- Training Parameters ---------------- #
lr = 2e-4
max_iters = 100000

m = IT1D()
optimizer = torch.optim.AdamW(m.parameters(),lr=lr)

m.to(device)
if os.path.isfile(pth):
    m.load_state_dict(torch.load(pth))
else:
    print("No prev net available")

for steps in range(max_iters):
    xb,yb,pos = ds.get_batch('train')
    # xb= xb.to(device)
    # pos =pos.to(device)
    # yb = yb.to(device)
    # print(xb.cuda)
    logits, loss = m(xb,pos,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if(steps % 5000 == 0):
        print(f"Step: {steps}/{max_iters} Loss : {ds.estimate_loss(m)}")
    if(steps == max_iters - 1):
        print(f"Step: {steps}/{max_iters} Loss : {ds.estimate_loss(m)}")


torch.save(m.state_dict(),pth)

imgs = [m.generate().to("cpu") for _ in range(9)]
fig,axs = plt.subplots(3,3)
[axs[i//3][i%3].imshow(imgs[i],cmap='gray') for i in range(9)]

# print(imgs[0])
# x = torch.cat([imgs[0][:,14:],imgs[0][:,:14]],dim=-1)
# print(x)
# plt.imshow(x,cmap='gray')
plt.show()