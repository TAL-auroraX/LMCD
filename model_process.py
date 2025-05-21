from typing import Optional

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed_utils import DeepspeedStrategy
from transformers.trainer import get_scheduler
import time
from transformers import AutoModel, AutoConfig,AutoTokenizer, PreTrainedTokenizer,AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig

class PosLinear(nn.Linear):
    # NCDM Head
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class Cdllm_Causal(nn.Module):
    def __init__(
        self,
        pretrain_or_model,
        num_tokens,
        num_kc,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        config = self.model.config

        # # use lora
        peft_config = LoraConfig(
            r=16,  # LoRA的秩,一般取8-32
            lora_alpha=32,  # LoRA的缩放参数
            lora_dropout=0.1,  # Dropout概率
        )
        print("use lora!")
        self.model = get_peft_model(self.model, peft_config)
        self.model.resize_token_embeddings(num_tokens)

        # Unfreeze embedding layer
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                print("activate embeddibg layer")
                param.requires_grad = True
        # IRT head
        discrimination = "discrimination"
        diff = "diff"
        guess = "guess"
        setattr(self,"StuViews",nn.Linear(config.hidden_size, 1, bias=False)) 
        setattr(self, discrimination, nn.Linear(config.hidden_size, 1, bias=False))   
        setattr(self, diff, nn.Linear(config.hidden_size, 1, bias=False))  
        setattr(self, guess, nn.Linear(config.hidden_size, 1, bias=False))  

        # MIRT head
        setattr(self,"MIRT_thetas",nn.Linear(config.hidden_size, num_kc, bias=False)) 
        setattr(self, "MIRT_alphas", nn.Linear(config.hidden_size, num_kc, bias=False))
        setattr(self, "MIRT_betas", nn.Linear(config.hidden_size, 1, bias=False))
        setattr(self, "MIRT_guess", nn.Linear(config.hidden_size, 1, bias=False))
        
        # NCDM head
        setattr(self,"NCDM_thetas",nn.Linear(config.hidden_size, num_kc, bias=False))
        setattr(self, "NCDM_alphas", nn.Linear(config.hidden_size, num_kc, bias=False))
        setattr(self, "NCDM_betas", nn.Linear(config.hidden_size, 1, bias=False))

        layers = [
            PosLinear(num_kc, num_kc*4),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(num_kc*4, num_kc*2),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(num_kc*2, 1)

        ]  
        setattr(self, "NCDM_FNN", nn.Sequential(*layers))
        
    
    def irt3pl(self,theta, a, b, c, D=1.702):
        """
        c: 表示题目的猜测因子
        a: 表示题目的区度
        b: 表示题目的难度
        """
        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))
        
    def forward(
        self,
        user_id: torch.LongTensor,
        inputs_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        knowledge_emb = None,
        cd_head = "IRT",
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        outputs = self.model(inputs_ids, attention_mask=attention_mask, position_ids=position_ids,output_hidden_states=True)
        student_emb = outputs["hidden_states"][0][:,-1,:].squeeze(1) 

        student_view = getattr(self, "StuViews")(student_emb)
        last_hidden_states = outputs["hidden_states"][-1] # batch * seq * hiddensize
        diff_student_view = last_hidden_states[:,-1,:].squeeze(1) # student view
        question_view = last_hidden_states[:,-2,:] # view of question

        ########IRT head##########
        if cd_head == "IRT":
            # best
            diff_emb =  getattr(self, "diff")(diff_student_view)
            guess_emb =  getattr(self, "guess")(question_view)
            dis_emb = getattr(self, "discrimination")(question_view)

            # Clip values to prevent explosion
            diff_emb = torch.clamp(diff_emb, min=-10.0, max=10.0)
            dis_emb = torch.clamp(dis_emb, min=-10.0, max=10.0)
            student_view = torch.clamp(student_view, min=-10.0, max=10.0)
            dis_emb = F.softplus(dis_emb)
            guess_emb = torch.sigmoid(guess_emb) 
            irt_loss = self.irt3pl(student_view,dis_emb,diff_emb,guess_emb)
            return irt_loss
        ########MRT head##########
        if cd_head == "MIRT":
            mirt_theta = getattr(self, "MIRT_thetas")(student_emb)
            mirt_alpha = getattr(self, "MIRT_alphas")(question_view)
            mirt_beta = getattr(self, "MIRT_betas")(diff_student_view)
            mirt_guess = getattr(self, "MIRT_guess")(question_view)
            mirt_guess = torch.sigmoid(mirt_guess)
            mirt_pred = torch.sigmoid((mirt_alpha * mirt_theta).sum(dim=1, keepdim=True) - mirt_beta)   
            pred = mirt_guess + (1 - mirt_guess) * mirt_pred
            return pred
        ########NCDM head##########
        if cd_head == "NCDM":
            # bulid knowledge embedding
            ncdm_theta = torch.sigmoid(getattr(self, "NCDM_thetas")(student_emb))
            ncdm_alpha = torch.sigmoid(getattr(self, "NCDM_alphas")(diff_student_view)) #diff
            ncdm_beta = torch.sigmoid(getattr(self, "NCDM_betas")(question_view))
            ncdm_output = ncdm_beta * (ncdm_theta - ncdm_alpha) * torch.tensor(knowledge_emb).to(ncdm_beta.device)
            ncdm_output = torch.sigmoid(getattr(self, "NCDM_FNN")(ncdm_output))           
            return ncdm_output

def get_cdm_with_causal(llm_model_path,additional_tokens,num_kc,cdm_head,padding_side="left"):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    # add student token
    tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})    
    tokenizer.padding_side = padding_side
    assert cdm_head in ["IRT", "MIRT","NCDM"], "CDM Head Error"
    if cdm_head == "MIRT":
        num_kc = 4
    model = Cdllm_Causal(pretrain_or_model = llm_model_path,num_tokens = len(tokenizer),num_kc=num_kc)
    return tokenizer,model
