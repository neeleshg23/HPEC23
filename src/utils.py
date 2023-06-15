import dvc.api

from models.d import DenseNetStudent, DenseNetTeacher
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
        dim = params["model"][f"stu_{option}"]["dim"]
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
    # elif option == "m":
    #     model = tch_m

def select_tch(option):
    params = dvc.api.params_show()
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    if option == "d":
        channels = params["model"][f"tch_{option}"]["channels"]
        return DenseNetTeacher(num_classes, channels)
    elif option == "r":
        dim = params["model"][f"tch_{option}"]["dim"]
        channels = params["model"][f"tch_{option}"]["channels"]
        return resnet50(num_classes, channels)
    elif option == "v":
        dim = params["model"][f"tch_{option}"]["dim"]
        depth = params["model"][f"tch_{option}"]["depth"]
        heads = params["model"][f"tch_{option}"]["heads"]
        mlp_dim = params["model"][f"tch_{option}"]["mlp-dim"]
        channels = params["model"][f"tch_{option}"]["channels"]
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

def select_cluster(option):
    pass