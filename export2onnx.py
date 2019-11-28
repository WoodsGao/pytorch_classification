import torch
from models import EfficientNet
from utils.modules.nn import WSConv2d

dummy_input = torch.rand([1, 3, 64, 64])
model = EfficientNet(20)
weights = torch.load('best.pt', map_location='cpu')['model']
model.load_state_dict(weights)
model.eval()
model.fuse()
torch.onnx.export(model, dummy_input, 'best.onnx')