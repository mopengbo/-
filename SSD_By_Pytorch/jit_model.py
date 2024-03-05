import torch
from model import TinySSD
from d2l import torch as d2l
from torch.utils.mobile_optimizer import optimize_for_mobile

model = TinySSD(num_classes=2)
state = torch.load('./models/SSD_state_220.pt',map_location='cpu')#205 220
model.load_state_dict(state['model'])
model.eval()
device = d2l.try_gpu()
example = torch.rand(1, 3, 512, 512).to(device)
jit_model = torch.jit.trace(model,example)
torch.jit.save(jit_model,"jit_model")