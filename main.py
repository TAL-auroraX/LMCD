from typing import Optional
import yaml
import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import os
from deepspeed_utils import DeepspeedStrategy
from transformers.trainer import get_scheduler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoConfig,AutoTokenizer, PreTrainedTokenizer,AutoModelForCausalLM
from model_process import get_cdm_with_causal
from data_process import get_userdata
from sklearn.metrics import roc_auc_score, accuracy_score,mean_squared_error, mean_absolute_error
from utils import init_wandb,freeze_llm,check,modify_tensor,one_hot_encoding_dict,unique_agg

#################################################
with open("config/ds_config.json","r") as f:
    # deepspeed config
    ds_config = json.load(f)

with open("config/train_config.yaml","r") as f:
    # train config
    train_config = yaml.safe_load(f)

# bulid deepspeed strategy
strategy = DeepspeedStrategy(
        seed = ds_config["seed"],
        max_norm = ds_config["max_norm"],
        micro_train_batch_size = ds_config["micro_train_batch_size"],
        train_batch_size = ds_config["train_batch_size"],
        zero_stage = ds_config["zero_stage"],
        bf16 = ds_config["bf16"],
        use_adam_offload = ds_config["adam_offload"],
    )
strategy.setup_distributed()

use_wandb = train_config["use_wandb"]
if strategy.is_rank_0() and use_wandb:
    # if use wandb
    with open("config/wandb_config.json","r") as f:
        wandb_config = json.load(f)
    init_wandb(wandb_config)

# process data
fold_id = train_config["fold_id"]
task_name = train_config["task_name"]
use_dataset = train_config["use_dataset"]
if task_name == "exercise_cold_start":
    if use_dataset == "NIPS34":
        data_path = train_config["nips34_exercise_cold_start"]
        train_data_path = data_path + "train_{}.csv".format(fold_id)
        oracle_data_path = data_path + "oracle_{}.csv".format(fold_id)
        test_data_path = data_path + "test_{}.csv".format(fold_id)
        question2text_path = "data/kcs_discription/nips34_kcs_discription.json"
    elif use_dataset == "XES3G5M":
        data_path = train_config["xes3g5m_exercise_cold_start"]
        train_data_path = data_path + "train_{}.csv".format(fold_id)
        oracle_data_path = data_path + "oracle_{}.csv".format(fold_id)
        test_data_path = data_path + "test_{}.csv".format(fold_id)
        question2text_path = "data/kcs_discription/xes_kcs_discription.json"
    else:
        raise ValueError("dataset name error")
elif task_name == "cross_domain_cold_start":
    if use_dataset == "XES3G5M":
        raise ValueError("cross domain cold start not support XES3G5M")
    data_path = train_config["cross_domain_cold_start"]
    target_domain = train_config["target_domain"]
    train_data_path = data_path + "{}_train.csv".format(target_domain)
    oracle_data_path = data_path + "{}_oracle.csv".format(target_domain)
    test_data_path = data_path + "{}_test.csv".format(target_domain)
    question2text_path = "data/kcs_discription/nips34_kcs_discription.json"
else:
    raise ValueError("task name error")

# load log
train_data = pd.read_csv(train_data_path)
oracle_data = pd.read_csv(oracle_data_path)
test_data = pd.read_csv(test_data_path)
# load kcs discription
with open(question2text_path, "r") as f:
    q2text = json.load(f)

unique_userids = sorted(train_data['UserId'].unique())
userid_map = {old_id: new_id for new_id, old_id in enumerate(unique_userids)}
train_data['UserId'] = train_data['UserId'].map(userid_map)
oracle_data['UserId'] = oracle_data['UserId'].map(userid_map)
test_data['UserId'] = test_data['UserId'].map(userid_map)

# resize uids,uid & add student token
all_uids = train_data["UserId"].tolist()
add_tokens = ["<Stu{}EMB>".format(it) for it in list(set(train_data["UserId"].tolist()))]

data_all = pd.concat([train_data, oracle_data, test_data], ignore_index=True)
kc_list = list(set(data_all["Kc"].tolist()))
num_kc = len(kc_list) + 1
all_one_hots = one_hot_encoding_dict(kc_list)
data_all['onehot'] = data_all['Kc'].map(all_one_hots)
q_to_onehot = data_all.groupby('QuestionId')['onehot'].apply(unique_agg).to_dict() # prepare for NCDM

# load model
cdm_head = train_config["cd_head_name"]
padding_side = train_config["padding_side"]
llm_model_path = train_config["llm_model_path"]
tokenizer,model = get_cdm_with_causal(llm_model_path,add_tokens,num_kc,cdm_head,padding_side = padding_side)
user_dataloader = get_userdata(train_data,q2text,tokenizer) # dataloader for train
test_user_dataloader = get_userdata(test_data,q2text,tokenizer) # dataloader for test
train_dataloader = strategy.setup_dataloader(
    replay_buffer=user_dataloader,
    batch_size=ds_config["micro_train_batch_size"],
    pin_memory=True,
    shuffle=True,
)
test_dataloader = strategy.setup_dataloader(
    replay_buffer=test_user_dataloader,
    batch_size=ds_config["micro_train_batch_size"],
    pin_memory=True,
    shuffle=True,
)

epochs = train_config["epochs"]
lr = train_config["lr"]
system_input = train_config["system_input"]
num_training_steps = len(train_dataloader)*epochs # train steps
warmup_steps = len(train_dataloader)*2 # warmup steps
optim = strategy.create_optimizer(model, lr=lr, betas=(0.9, 0.95),weight_decay=0.0)
scheduler = get_scheduler(
    name="linear",       
    optimizer=optim,    
    num_warmup_steps=warmup_steps,    
    num_training_steps=num_training_steps
)
(model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
bce_loss = nn.BCELoss() 
model.train()

# do eval
def do_evalution(model,test_dataloader,task = "test",padding_side = "left"):
    model.eval()
    y_pred = []
    y_true = []
    tokenizer.padding_side = "left"
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing Progress", unit="batch"):
            user_id, text_list,scores,items = batch
            final_text = []
            knowledge_emb = []
            for it in zip(user_id,text_list,scores,items):
                this_id = it[0]
                this_question = it[1]
                this_score = it[2]
                this_item = it[3]
                knowledge_emb.append(q_to_onehot[int(this_item)][0])
                this_discript = system_input + this_question + "\n<Stu{}EMB>".format(this_id)
                final_text.append(this_discript)
            all_input = tokenizer(final_text,return_tensors='pt',padding='longest')
            scores = scores.unsqueeze(1).to(torch.bfloat16).to(torch.cuda.current_device())
            input_ids = all_input["input_ids"].to(torch.cuda.current_device())
            input_attention = all_input["attention_mask"].to(torch.cuda.current_device())
            user_id = user_id.to(torch.cuda.current_device()) #userid 4
            cdm_output = model(user_id, input_ids, input_attention,knowledge_emb = knowledge_emb,cd_head = cdm_head)
            loss = bce_loss(cdm_output,scores)
            y_pred.extend(cdm_output.tolist())
            y_true.extend(scores.tolist())
        if task == "test":
            eval_log = {}
            eval_log["valid loss"] = loss.item()
            eval_log["acc"] = accuracy_score(y_true, np.array(y_pred) >= 0.5)
            eval_log["auc"] = roc_auc_score(y_true, y_pred)
            eval_log["mae"] = mean_absolute_error(y_true, y_pred)
            eval_log["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            all_eval_logs = strategy.all_reduce(eval_log)
            if strategy.is_rank_0() and use_wandb:
                wandb.log(all_eval_logs)
            print("test",all_eval_logs)
        else:
            eval_log = {}
            eval_log["val-acc"] = accuracy_score(y_true, np.array(y_pred) >= 0.5)
            eval_log["val-auc"] = roc_auc_score(y_true, y_pred)
            eval_log["val-rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            all_eval_logs = strategy.all_reduce(eval_log)
            if strategy.is_rank_0() and use_wandb:
                wandb.log(all_eval_logs)
            print("val:",all_eval_logs)
    tokenizer.padding_side = padding_side
    model.train()

# train loop
for epoch in range(epochs):
    ce_loss_list = []
    pov_loss_list = []

    step_bar = tqdm(
                range(len(train_dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not strategy.is_rank_0(),
                unit="batch"
            )
    for step, batch in enumerate(train_dataloader):
        user_id, text_list,scores,items = batch
        final_text = []
        knowledge_emb = []
        for it in zip(user_id,text_list,scores,items):
            this_id = it[0]
            this_question = it[1].strip()
            this_score = it[2]
            this_item = it[3]

            knowledge_emb.append(q_to_onehot[int(this_item)][0])
            this_discript = system_input + this_question + "\n<Stu{}EMB>".format(this_id)
            final_text.append(this_discript)
        all_input = tokenizer(final_text,return_tensors='pt',padding='longest')
        scores = scores.unsqueeze(1).to(torch.bfloat16).to(torch.cuda.current_device())
        input_ids = all_input["input_ids"].to(torch.cuda.current_device())
        input_attention = all_input["attention_mask"].to(torch.cuda.current_device())
        user_id = user_id.to(torch.cuda.current_device()) 
        cdm_output = model(user_id, input_ids, input_attention,knowledge_emb = knowledge_emb,cd_head = cdm_head)
        loss = bce_loss(cdm_output,scores)
        if strategy.is_rank_0():        
            ce_loss_list.append(loss.item())

        if strategy.is_rank_0() and step%10==0 and use_wandb:
            tmp = {}
            tmp["ce"] = float(np.mean(ce_loss_list))
            wandb.log(tmp)

        # backward
        strategy.backward(loss, model, optim)
        strategy.optimizer_step(optim, model, scheduler)
        step_bar.update(1)
        format_info = step_bar.format_dict
        if strategy.is_rank_0() and step%50==0: 
            current_step = format_info['n']       
            remaining_steps = format_info['total'] - format_info['n']
            remaining_time = (format_info['total'] - format_info['n']) / format_info['rate']
            step_info = f"Current step: {current_step}, Steps remaining: {remaining_steps}, Time needed: {remaining_time:.2f}seconds"
            print(step_info)
    do_evalution(model,test_dataloader,task="test",padding_side=padding_side)
    # save_model
    starategy.save_ckpt(model,"/save_model/")