import dvc.api

from models.d import DenseNetStudent, DenseNetTeacher
from models.l import LSTMNet
from models.m import MLPMixer
from models.r import resnet_tiny, resnet50
from models.v import TMAP

def select_stu(option):
    params = dvc.api.params_show()
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    if option == "d":
        channels = params["model"][f"stu_{option}"]["channels"]
        return DenseNetStudent(num_classes, channels)
    elif option == "r":
        channels = params["model"][f"stu_{option}"]["channels"]
        return resnet_tiny(num_classes, channels)
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
        vocab_size = params["model"][f"stu_{option}"]["vocab-size"]
        embed_dim = params["model"][f"stu_{option}"]["embed-dim"]
        hidden_dim = params["model"][f"stu_{option}"]["hidden-dim"]
        num_layers = params["model"][f"stu_{option}"]["num-layers"]
        return LSTMNet(vocab_size, embed_dim, hidden_dim, num_layers)
    
def select_tch(option):
    params = dvc.api.params_show()
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    if option == "d":
        channels = params["model"][f"tch_{option}"]["channels"]
        return DenseNetTeacher(num_classes, channels)
    elif option == "r":
        channels = params["model"][f"tch_{option}"]["channels"]
        return resnet50(num_classes, channels)
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
        vocab_size = params["model"][f"tch_{option}"]["vocab-size"]
        embed_dim = params["model"][f"tch_{option}"]["embed-dim"]
        hidden_dim = params["model"][f"tch_{option}"]["hidden-dim"]
        num_layers = params["model"][f"tch_{option}"]["num-layers"]
        return LSTMNet(vocab_size, embed_dim, hidden_dim, num_layers)