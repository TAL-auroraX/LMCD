import os
import wandb

def init_wandb(config):
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = config["wandb_dir_path"]
    os.environ["WANDB_CACHE_DIR"] =  config["wandb_dir_path"]
    os.environ["WANDB_CONFIG_DIR"] = config["wandb_dir_path"]
    os.environ["WANDB_API_KEY"] = config["wandb_key"]
    wandb.login(host=config["wandb_host"])
    wandb.init(
        project= config["wandb_project"],
        entity= config["wandb_entity"],
        name=config["lab_tag"],
        job_type="training",
    )

def check(model):
    """
    check acitvate weight
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is unfrozen")

def freeze_llm(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.StuViews.weight.requires_grad = True
    model.model.embed_tokens.weight.requires_grad = True
    model.discrimination.weight.requires_grad = True
    #model.proficiency.weight.requires_grad = True
    model.guess.weight.requires_grad = True
    model.diff.weight.requires_grad = True
    return model

def modify_tensor(indices, labels_indexs,tensor):
    new_tensor = tensor.clone()
    for i, index in enumerate(indices):
        new_tensor[i, :index] = -100
        new_tensor[i, index+labels_indexs[i]:] = -100
    return new_tensor


def one_hot_encoding_dict(lst):
    unique_elements = sorted(set(lst))
    one_hot_dict = {
        element: [int(element == unique) for unique in unique_elements] + [1]
        for element in unique_elements
    }
    return one_hot_dict


# 定义一个去重的方法
def unique_agg(x):
    return list(map(list, {tuple(i) for i in x}))