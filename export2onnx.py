import torch
from pytorch_modules.backbones.resnet import resnet18
import cv2

dummy_input = torch.rand([1, 3, 64, 64])
model = resnet18(num_classes=3)
weights = torch.load('weights/plate.pt', map_location='cpu')['model']
model.load_state_dict(weights)
model.eval()
torch.onnx.export(model, dummy_input, 'best.onnx')
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setInput(dummy_input.numpy())
net.forward()
