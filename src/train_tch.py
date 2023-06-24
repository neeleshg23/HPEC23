import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

import dvc.api
from dvclive import Live

from utils import select_tch

torch.manual_seed(100)

model = None
optimizer = None
scheduler = None
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        output = sigmoid(model(data))
        loss = F.binary_cross_entropy(output, target, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = sigmoid(model(data))
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, loading, model_save_path, train_loader, test_loader, tch, live):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        print("-------------Model Loaded------------")
        
    best_loss=0
    early_stop = early_stop
    curr_early_stop = early_stop
    for epoch in range(epochs):

        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        print((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        live.log_metric(f"train_tch_{tch}/train_loss", train_loss)
        live.log_metric(f"train_tch_{tch}/test_loss", test_loss)

        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            print("-------- Save Best Model! --------")
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            print("Early Stop Left: {}".format(curr_early_stop))
        if curr_early_stop == 0:
            print("-------- Early Stop! --------")
            break

        live.next_step()

    
def main():
    global model
    global optimizer
    global scheduler

    params = dvc.api.params_show()

    teacher_models = params["teacher"]["models"]
    gpu_id = params["system"]["gpu-id"]
    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    
    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    for tch, option in enumerate(teacher_models, start=1):
        model = select_tch(option)
        print(summary(model))
        model_save_path = os.path.join(model_dir, f"teacher_{tch}.pth")
        
        print("Loading data for tch:", tch)
        train_loader = torch.load(os.path.join(processed_dir, f"train_loader_{tch}.pt"))
        test_loader = torch.load(os.path.join(processed_dir, f"test_loader_{tch}.pt"))

        print("Data loaded successfully for tch:", tch)
                                                                                                                
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        loading = False

        with Live(dir="res", resume=True) as live:
            live.step = 1
            run_epoch(epochs, early_stop, loading, model_save_path, train_loader, test_loader, tch, live)

if __name__ == "__main__":
    main()