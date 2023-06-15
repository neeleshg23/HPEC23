import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

import dvc.api
from dvclive import Live

from utils import select_stu, select_tch

torch.manual_seed(100)

model = None
teacher_model = None
optimizer = None
scheduler = None
device = None
alpha = None
Temperature = None

soft_loss = nn.KLDivLoss(reduction="mean", log_target=True)
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_preds = model(data)
        
        with torch.no_grad():
            teacher_preds = teacher_model(data)
        
        student_loss = F.binary_cross_entropy(sigmoid(student_preds), target, reduction='mean')

        x_t_sig = sigmoid(teacher_preds / Temperature).reshape(-1)
        x_s_sig = sigmoid(student_preds / Temperature).reshape(-1)

        x_t_p = torch.stack((x_t_sig, 1 - x_t_sig), dim=1)
        x_s_p = torch.stack((x_s_sig, 1 - x_s_sig), dim=1)

        distillation_loss = soft_loss(x_s_p.log(), x_t_p.log())
        loss = alpha * student_loss + (1 - alpha) * distillation_loss

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
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, loading, tch_model_save_path, model_save_path, train_loader, test_loader, live):
    teacher_model.load_state_dict(torch.load(tch_model_save_path))
    print("-------------Teacher Model Loaded------------")

    
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        print("-------------Model Loaded------------")
        
    best_loss=0
    early_stop = early_stop
    curr_early_stop = early_stop

    for epoch in range(epochs):
        train_loss=train(epoch, train_loader, model_save_path)
        test_loss=test(test_loader)
        print((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        live.log_metric("train_stu/train_loss", train_loss)
        live.log_metric("train_stu/test_loss", test_loss)

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
    global teacher_model
    global optimizer
    global scheduler
    global device
    global alpha
    global Temperature

    params = dvc.api.params_show()

    stu_option = params["student"]["model"]
    tch_option = params["teacher"]["model"]

    gpu_id = params["system"]["gpu-id"]
    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]
    alpha = params["train"]["alpha"]
    Temperature = params["train"]["temperature"]

    teacher_model = select_tch(tch_option)
    model = select_stu(stu_option)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    tch_model_save_path = os.path.join(model_dir, "teacher_1.pth")
    model_save_path = os.path.join(model_dir, "student.pth")

    train_loader = torch.load(os.path.join(processed_dir, "train_loader_1"))
    test_loader = torch.load(os.path.join(processed_dir, "test_loader_1"))

    model = model.to(device)
    teacher_model = teacher_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loading = False

    with Live(dir="res", resume=True) as live:
        live.step = 1
        run_epoch(epochs, early_stop, loading, tch_model_save_path, model_save_path, train_loader, test_loader, live)

if __name__ == "__main__":
    main()