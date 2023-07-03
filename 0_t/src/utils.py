import os
import yaml
from models.d import DenseNetStudent, DenseNetTeacher
from models.l import LSTMModel
from models.m import MLPMixer
from models.r import resnet_tiny, resnet50
from models.v import TMAP

def select_stu(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    if option == "d":
        channels = params["model"][f"stu_{option}"]["channels"]
        return DenseNetStudent(num_classes, channels)
    elif option == "r":
        channels = params["model"][f"stu_{option}"]["channels"]
        dim = params["model"][f"stu_{option}"]["dim"]
        return resnet_tiny(num_classes, channels, dim)
    elif option == "v":
        dim = params["model"][f"stu_{option}"]["dim"]
        depth = params["model"][f"stu_{option}"]["depth"]
        heads = params["model"][f"stu_{option}"]["heads"]
        mlp_dim = params["model"][f"stu_{option}"]["mlp-dim"]
        channels = params["model"][f"stu_{option}"]["channels"]
        return TMAP(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=mlp_dim
        )
    elif option == "m":
        channels = params["model"][f"stu_{option}"]["channels"]
        dim = params["model"][f"stu_{option}"]["dim"]
        depth = params["model"][f"stu_{option}"]["depth"]
        return MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size[1],
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
    elif option == "l":
        input_dim = params["model"][f"stu_{option}"]["input-dim"]
        hidden_dim = params["model"][f"stu_{option}"]["hidden-dim"]
        layer_dim = params["model"][f"stu_{option}"]["layer-dim"]
        output_dim = params["model"][f"stu_{option}"]["output-dim"]
        
        return LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    
def select_tch(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    if option == "d":
        channels = params["model"][f"tch_{option}"]["channels"]
        return DenseNetTeacher(num_classes, channels)
    elif option == "r":
        channels = params["model"][f"tch_{option}"]["channels"]
        dim = params["model"][f"tch_{option}"]["dim"]
        return resnet50(num_classes, channels, dim)
    elif option == "v":
        dim = params["model"][f"tch_{option}"]["dim"]
        depth = params["model"][f"tch_{option}"]["depth"]
        heads = params["model"][f"tch_{option}"]["heads"]
        mlp_dim = params["model"][f"tch_{option}"]["mlp-dim"]
        channels = params["model"][f"tch_{option}"]["channels"]
        return TMAP(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            channels = channels,
            dim_head = mlp_dim
        )
    elif option == "m":
        channels = params["model"][f"tch_{option}"]["channels"]
        dim = params["model"][f"tch_{option}"]["dim"]
        depth = params["model"][f"tch_{option}"]["depth"]
        return MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size[1],
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
    elif option == "l":
        input_dim = params["model"][f"tch_{option}"]["input-dim"]
        hidden_dim = params["model"][f"tch_{option}"]["hidden-dim"]
        layer_dim = params["model"][f"tch_{option}"]["layer-dim"]
        output_dim = params["model"][f"tch_{option}"]["output-dim"]
        
        return LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
