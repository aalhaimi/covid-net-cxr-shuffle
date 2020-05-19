# ---------------- COVIDNet-CXR-Shuffle  -------------------- # 
import time
import torch
import torchvision
import numpy as np 
import utils 
import torch.utils.data
from sklearn.metrics import confusion_matrix, accuracy_score

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Using {device} for prediction')


# ----------  INPUTS ----------
model_path = 'models/covidnet-cxr-shuffle-e35.pth' 
data_path_tst = 'data/covidx2_test'


# ---------- DATA LOADING ----------
trans = torchvision.transforms.Compose([
        utils.RemoveScanInfo(),
        torchvision.transforms.Resize((224,224)),
        utils.HistogramNorm(),
        torchvision.transforms.ToTensor(),
    ])

tst_set = torchvision.datasets.ImageFolder(data_path_tst, transform = trans)
tst = utils.load_alldataset(tst_set)
tst[1][:].shape
x = tst[0][:]
y = tst[1][:]
subtitle = 'COVIDx2 (tst)'


# ---------- DL MODEL ARCH ----------
model = torchvision.models.shufflenet_v2_x1_0().float().to(device)
model.fc = torch.nn.Linear(1024, 3) 


## --- RESTORE CHECKPOINTS ----
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


# ---------- PREDICTIONS ----------
def predict(x, y):
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        pred = model(x)
        label = y.to(device)
        pred = pred.to(device)
        return (label, pred)
 
start = time.time()
val_label, val_pred = predict(x.to(device), y.to(device))
val_label, val_pred = val_label.cpu(), val_pred.cpu()
val_pred = [ np.argmax(t) for t in val_pred ]


## ---------- CONFUSION MATRIX ----------
cm = confusion_matrix(val_label, val_pred)
acc = accuracy_score(val_label, val_pred)
print(f'[INFO] Confusion Matrix of {subtitle}')
print('class 0: COVID-19, class 1: normal, class 2: pneumonia')
print(cm)
print(f'Overall Accuracy: {round(acc * 100, 3)}%')

print(f'[INFO] Predicted {x.shape[0]} images in {round(time.time() - start, 4)} seconds on {device}')
print(f'[INFO] Prediction is complete.')