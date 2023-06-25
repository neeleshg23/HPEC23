import os
import sys
import warnings

from validate_stu import run_val
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
teacher_models = None
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

def run_epoch(epochs, early_stop, loading, teacher_base_save_path, model_save_path, train_loaders, test_loaders, df_test_stu, test_loader_stu, app, live):
    best_loss = 0
    early_stop = early_stop
    curr_early_stop = early_stop
    num_teachers = len(train_loaders)

    if loading:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("-------------Model Loaded------------")

    for epoch in range(epochs):
        losses_output = f"Epoch {epoch:2.0f}: "
        
        for i in range(num_teachers):
            tch_model_path = f"{teacher_base_save_path}_{i+1}.pth"
            teacher_model = teacher_models[i].to(device)
            teacher_model.load_state_dict(torch.load(tch_model_path, map_location=device))

            train_loss = train(epoch, train_loaders[i], model_save_path, teacher_model)

            test_loss = test(test_loaders[i])

            losses_output += f"T{i+1}: TrainL={train_loss:.5f}, TestL={test_loss:.5f}; "
            
            live.log_metric(f"train_stu/train_loss_{i+1}", train_loss)
            live.log_metric(f"train_stu/test_loss_{i+1}", test_loss)

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
        
        print("-- -- START EPOCH -- --")
        print(losses_output)

        res = run_val(test_loader_stu, df_test_stu, app, model_save_path)
        
        live.log_metric("train_stu/opt_threshold", res["opt_th"][0])
        live.log_metric("train_stu/precision", res["p"][0])
        live.log_metric("train_stu/precision_5", res["p_5"][0])
        live.log_metric("train_stu/recall", res["r"][0])
        live.log_metric("train_stu/recall_5", res["r_5"][0])
        live.log_metric("train_stu/accuracy", res["f1"][0])
        live.log_metric("train_stu/accuracy_5", res["f1_5"][0])

        print(f"Epoch {epoch:2.0f} Val: opt_threshold={res['opt_th'][0]}, precision={res['p'][0]}, precision_5={res['p_5'][0]}, recall={res['r'][0]}, recall_5={res['r_5'][0]}, accuracy={res['f1'][0]}, accuracy_5={res['f1_5'][0]}")
        print("-- -- END EPOCH -- --")


        live.next_step()


def main():
    global model
    global teacher_models
    global optimizer
    global scheduler
    global device
    global alpha
    global Temperature

    params = dvc.api.params_show()

    app = params["apps"]["app"]

    stu_option = params["student"]["model"]
    teacher_models = params["teacher"]["models"]

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

    model = select_stu(stu_option)
    print(summary(model))
    model_save_path = os.path.join(model_dir, "student.pth")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    df_test_stu = torch.load(os.path.join(processed_dir, "df_test_stu.pt"))
    test_loader_stu = torch.load(os.path.join(processed_dir, "test_loader_stu.pt"))

    train_loaders = []
    test_loaders = []
    teacher_models = [select_tch(option) for option in teacher_models]
    for i, option in enumerate(teacher_models, start=1):
        train_loader = torch.load(os.path.join(processed_dir, f"train_loader_{i}.pt"))
        test_loader = torch.load(os.path.join(processed_dir, f"test_loader_{i}.pt"))
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    teacher_base_save_path = os.path.join(model_dir, "teacher")
    model_save_path = os.path.join(model_dir, "student.pth")


    loading = False
    with Live(dir="res", resume=True) as live:
        live.step = 1
        run_epoch(epochs, early_stop, loading, teacher_base_save_path, model_save_path, train_loaders, test_loaders, df_test_stu, test_loader_stu, app, live)


if __name__ == "__main__":
    main()