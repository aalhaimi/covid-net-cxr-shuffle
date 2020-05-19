# ---------------- COVIDNet-CXR-Shuffle -------------------- # 
import torch
import torchvision
import numpy as np 
import utils 
import torch.utils.data

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Using {device} to train the model')


# ----------  INPUTS ----------
model_path = 'models/covidnet-cxr-shuffle-model.pth' 
data_path = 'data/trainset'

epochs = 20
batch_size = 32

# inits
train_losses = []
val_losses = []
val_accs = []
cur_epoch = 0
best_acc = 0


# ---------- DATA LOADING AND AUGMENTATIONS ----------
trn_trans = torchvision.transforms.Compose([
    utils.RemoveScanInfo(),
    torchvision.transforms.Resize((224,224)),
    utils.HistogramNorm(),
    torchvision.transforms.RandomOrder([
        torchvision.transforms.RandomAffine(degrees=(-45,45), translate=(0.1,0.1), scale=(0.9,1.1)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        ]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(p=0.2, scale=(0.003, 0.03))
    ])

val_trans = torchvision.transforms.Compose([
        utils.RemoveScanInfo(),
        torchvision.transforms.Resize((224,224)),
        utils.HistogramNorm(),
        torchvision.transforms.ToTensor(),
    ])

train_set = torchvision.datasets.ImageFolder(data_path + '/trnb/', transform = trn_trans)
val_set = torchvision.datasets.ImageFolder(data_path + '/valb/', transform = val_trans)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)


# ---------- DL MODEL ARCH ----------
model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model.fc = torch.nn.Linear(1024, 3) 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
adjustLR = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, mode='exp_range', cycle_momentum=False)
ce_loss = torch.nn.CrossEntropyLoss()

# --- RESTORE CHECKPOINTS ----
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# cur_epoch = checkpoint['epochs'] - 1
# best_acc = checkpoint['best_acc']


# --- TRAIN FOR-LOOP ----
for e in range(epochs):
    e += cur_epoch + 1
    running_loss = 0 
    for i, (inputs, targets) in enumerate(train_loader):
        model = model.to(device)
        model = model.train() 
        x = inputs.to(device)   
        y = targets.to(device)   
        optimizer.zero_grad()
        outputs = model(x)
        loss = ce_loss(outputs, y)
        loss.backward()
        optimizer.step()
        adjustLR.step()
        running_loss += loss.item()
    else:
        val_loss = 0
        val_acc = 0
        batch_correct = 0
        batch_total = 0
        with torch.no_grad():
            model = model.eval()
            for i, (inputs, targets) in enumerate(val_loader):
                x = inputs.to(device)      
                y = targets.to(device)   
                outputs = model(x)
                loss = ce_loss(outputs, y)
                val_loss += loss.item()
                batch_total += y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                batch_correct += (predicted == y).sum().item()
                val_acc = batch_correct / batch_total
        val_accs.append( val_acc )
        val_losses.append(val_loss / len(val_loader))
        train_losses.append(running_loss / len(train_loader) )

        if val_accs[-1] >= best_acc:
            best_acc = val_accs[-1]
            torch.save({
                    'model': model.train(),
                    'epochs': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc}, model_path)

        print ('Epochs: {}/{} '.format(e, epochs),
                'Train Loss: {:.7f} '.format(train_losses[-1]),
                'Val Loss: {:.7f} '.format(val_losses[-1]),
                'Val Acc: {:.7f}'.format(val_accs[-1]),
                'Best Val Acc: {:.7f}'.format(best_acc))

print('[INFO] Model training is complete.')
