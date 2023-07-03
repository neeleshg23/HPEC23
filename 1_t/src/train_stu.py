import csv
import os
import sys
import warnings
from data_loader import init_dataloader

import yaml

from validate_stu import run_val
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

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

def train(ep, train_loader, model_save_path, teacher_model):
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
        
        test_loss /= len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, loading, teacher_model_save_path, model_save_path, train_loader, test_loader, df_test, app, tsv_path, option, gpu_id):
    best_loss = 0
    early_stop = early_stop
    curr_early_stop = early_stop

    metrics_data = []

    teacher_model = select_tch(option)

    if loading:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("-------------Model Loaded------------")

    for epoch in range(epochs):
        losses_output = f"Epoch {epoch:2.0f}: "

        print(f"-------------Start Epoch {epoch}-------------")
        
        teacher_model = teacher_model.to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_save_path, map_location=device))

        train_loss = train(epoch, train_loader, model_save_path, teacher_model)
        test_loss = test(test_loader)
        
        losses_output += f"T1: TrainL={train_loss:.5f}, TestL={test_loss:.5f}; "
            
        if epoch == 0:
            best_loss = test_loss
        if test_loss <= best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss = test_loss
            losses_output += "Best Model Saved!"
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            losses_output += f"Early Stop Left: {curr_early_stop}!"
        
        if curr_early_stop == 0:
            losses_output += "Early Stop Triggered!"
        
        print(losses_output)

        res = run_val(test_loader, df_test, app, model_save_path, option, gpu_id)

        metrics_data.append([epoch+1, train_loss, test_loss, res["opt_th"][0], res["p"][0], res["r"][0], res["f1"][0], res["p_5"][0], res["r_5"][0], res["f1_5"][0]])
        print(f"Epoch {epoch+1:2.0f} Val: opt_threshold={res['opt_th'][0]}, precision={res['p'][0]}, precision_5={res['p_5'][0]}, recall={res['r'][0]}, recall_5={res['r_5'][0]}, accuracy={res['f1'][0]}, accuracy_5={res['f1_5'][0]}")
        
    with open(tsv_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Opt_Th', 'P', 'R', 'F1', 'P_5', 'R_5', 'F1_5'])
        writer.writerows(metrics_data)

def main():
    global model
    global teacher_model
    global optimizer
    global scheduler
    global device
    global alpha
    global Temperature

    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)

    app = sys.argv[1]
    app_name = app[:-7]

    option = sys.argv[2]
    tch_option = sys.argv[2]
    gpu_id = sys.argv[3]
    init_dataloader(gpu_id)

    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    results_dir = params["system"]["res"]

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]
    alpha = params["train"]["alpha"]
    Temperature = params["train"]["temperature"]

    model = select_stu(option)
    print(summary(model))

    teacher_model_save_path = os.path.join(model_dir, f"{app_name}.teacher.{tch_option}.pth")
    model_save_path = os.path.join(model_dir, f"{app_name}.student.{option}.pth")
    tsv_path = os.path.join(results_dir, f"{app_name}.student.{option}.tsv")
    

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    df_test = torch.load(os.path.join(processed_dir, f"{app_name}.df.pt"))
    train_loader = torch.load(os.path.join(processed_dir, f"{app_name}.train.pt"))
    test_loader = torch.load(os.path.join(processed_dir, f"{app_name}.test.pt"))

    teacher_model = select_tch(tch_option)

    loading = False
    run_epoch(epochs, early_stop, loading, teacher_model_save_path, model_save_path, train_loader, test_loader, df_test, app, tsv_path, option, gpu_id)


if __name__ == "__main__":
    main()
