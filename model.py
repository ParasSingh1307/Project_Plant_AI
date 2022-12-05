import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io


class Plant_Disease_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)
    
    def forward(self,xb):
        out = self.network(xb)
        return out


transform = transforms.Compose(
    [transforms.Resize(size=128),
     transforms.ToTensor()])

num_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy']


model = Plant_Disease_Model()
model.load_state_dict(torch.load(
    'C:\\Users\\PARAS SINGH\\Desktop\\Project\\plantDisease-resnet50.pth', map_location=torch.device('cpu')))
model.eval()


def predict_image(img):
    img_pil = Image.open(io.BytesIO(img))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]