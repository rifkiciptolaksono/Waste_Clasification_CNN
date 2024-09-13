import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torchvision.datasets as datasets
import os

from model import ResNet, BasicBlock

# medefinisikan data gambar 
images_dir = 'data_/gambar/class'
train_dir = 'data_/gambar/train'
test_dir = 'data_/gambar/test'

# membuat class gambar mejadi label
classes = os.listdir(images_dir)

train_data = datasets.ImageFolder(train_dir)
test_data = datasets.ImageFolder(test_dir)

classes = test_data.classes

OUTPUT_DIM = len(test_data.classes)

# mendefiniskan model resnet34 sesuai dengan model yang telah dilatih sebelumnya 
def resnet34_config(OUTPUT_DIM):
    config = (BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512])
    return ResNet(config, OUTPUT_DIM)

model = resnet34_config(OUTPUT_DIM) 

# menginputkan model resnet34 yang telah dilatih dan di save sebelumnya
model_path = 'model_final/resnet34.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

# mendefinisikan fungsi transform gambar untuk menyesuiakan dengan ukuran gambar pada model yang telah dilatih sebelumnya 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7044, 0.6648, 0.6648], std=[0.1715, 0.2011, 0.2263]),
])


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mengubah gambar dari format BGR ke RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb) 

    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0) 

    # fungsi untuk melakukan prediksi menggunakan model resnet34
    with torch.no_grad():
        outputs, _ = model(img_tensor)  

    probabilities = F.softmax(outputs, dim=1)
    
    # melakukan prediksi sesuai dengan label atau kelas
    _, predicted = torch.max(outputs, 1)
    label = classes [predicted.item()]

    prob = probabilities[0][predicted.item()].item()

    # tampilan pada camera
    text = f"{label}: {prob:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
