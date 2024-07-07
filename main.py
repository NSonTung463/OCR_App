from models.model import *
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(1, 62)
model = torch.load('weights/emnist-resnet18-full.pth', map_location=device)
model = model.to(device)
model.eval()
# Predict sequence of characters
sequence_image_path = "predict/sequence_image.png"
predictions,final_predict = predict_sequence(model, sequence_image_path,device)
display_prediction(sequence_image_path, predictions)