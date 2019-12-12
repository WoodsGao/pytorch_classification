import torch
from utils.models import EfficientNetGCM
import cv2

dummy_input = torch.rand([1, 3, 64, 64])
model = EfficientNetGCM(20)
weights = torch.load('best.pt', map_location='cpu')['model']
model.load_state_dict(weights)
model.eval()
model.fuse()
torch.onnx.export(model, dummy_input, 'best.onnx')
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setInput(dummy_input.numpy())
net.forward()
