import sys
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import torch
import random
from torch.utils.data import TensorDataset, DataLoader,Dataset,Sampler
from collections import defaultdict
from sklearn.model_selection import train_test_split

class UserDataset(Dataset):
    def __init__(self, data,tokenizer,q2info):
        self.data = data
        self.q2info = q2info
        self.tokenizer = tokenizer
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scores = self.data[idx]["score"]
        items =  str(int(self.data[idx]["items"]))
        userid =  self.data[idx]["u_id"]
        return userid,self.q2info[items],scores,items

def bulid_prompt(q2text,train_data):
    def dict_to_string(option_dict):
        result = ""
        for key, value in option_dict.items():
            result += f"{key}: {value}\n"
        return result

    q2info = {}
    for k,v in q2text.items():
        q_info = v.strip()
        q_all_info = "## Question Information\n### Question:\n{}\n## Student Token".format(q_info)
        q2info[k] = q_all_info

    all_dict = {}
    start = 0
    for index,row in train_data.iterrows():
        tmp = {}
        items = row["QuestionId"]
        scores = row["IsCorrect"]
        users = row["UserId"]

        tmp["u_id"] = users
        tmp["items"] = items
        tmp["score"] = scores
        all_dict[index] = tmp

    return q2info,all_dict

def get_userdata(train_data,q2text,tokenizer):
    q2info,all_dict = bulid_prompt(q2text,train_data)
    userdata = UserDataset(all_dict,tokenizer,q2info)
    return userdata

if __name__=="__main__":
    pass