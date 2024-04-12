import gc
import os
import pickle
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


random_state = 47

torch.manual_seed(random_state)
np.random.seed(random_state)
torch.cuda.manual_seed(random_state)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "cointegrated/rubert-tiny2"
batch_size = 64


class CustomVacancyDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        device: torch.device,
        text: str,
        target: str,
        chunk_size: int = 512,
        mode: Optional[Literal["train", "valid", "test"]] = None
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.text = text
        self.target = target
        self.chunk_size = chunk_size
        self.mode = mode

    def __len__(self):
        return len(self.data)

    @staticmethod
    def chunker(tokens: Dict[str, torch.Tensor], chunk_size: int = 512) -> List[str]:
        input_ids = list(tokens["input_ids"][0].split(chunk_size - 2))
        attention_mask = list(tokens["attention_mask"][0].split(chunk_size - 2))

        cls_token_id = 2
        eos_token_id = 3

        for i in range(len(input_ids)):
            input_ids[i] = torch.cat([torch.Tensor([cls_token_id]), input_ids[i], torch.Tensor([eos_token_id])])
            attention_mask[i] = torch.cat([torch.Tensor([1]), attention_mask[i], torch.Tensor([1])])

            pad_len = chunk_size - len(input_ids[i])
            if pad_len > 0:
                input_ids[i]= torch.cat([input_ids[i], torch.Tensor([0]*pad_len)])
                attention_mask[i] = torch.cat([attention_mask[i], torch.Tensor([0]*pad_len)])
                
        tokens["input_ids"] = torch.stack(input_ids)
        tokens["attention_mask"] = torch.stack(attention_mask)
        
        return tokens
    
    def __getitem__(self, idx: int):
        tokens = self.tokenizer(self.data.iloc[idx][self.text], return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
        tokens = self.chunker(tokens, self.chunk_size)
        
        tokens["input_ids"] = tokens["input_ids"].to(device).long()
        tokens["attention_mask"] = tokens["attention_mask"].to(device).int()

        if self.mode == "test":
            return tokens

        label = self.data.iloc[idx][self.target]
        return tokens, torch.tensor(label)


class BertClassification(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

        self.seq_0 = nn.Sequential(
            nn.Linear(312, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
    def forward(self, inputs):
        x = self.bert(**inputs)
        x = (x["last_hidden_state"] * inputs["attention_mask"][:, :, None]).sum(dim=1) / inputs["attention_mask"][:, :, None].sum(dim=1)
        x = self.seq_0(torch.mean(x, dim=0))

        return x


def get_score(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec, micro_prec = precision_score(y_true, y_pred, average="macro"), precision_score(y_true, y_pred, average="micro")
    macro_rec = recall_score(y_true, y_pred, average="macro")
    
    return macro_f1, macro_prec, macro_rec, micro_prec


def train_fn(model, loss, data, optimizer):
    model.train()
    metrics = 0.0
    y_true, y_pred = [], []
    

    for i, batch in enumerate(tqdm(data, leave=False, desc="Train")):
        tokens_list, target = batch
        target = target.to(device)
        temp_outputs = []

        optimizer.zero_grad()
        for tokens in tokens_list:
            output = model(tokens)
            temp_outputs.append(output)
        temp_outputs = torch.stack(temp_outputs)
        
        loss_remains = loss(temp_outputs, target)
        loss_remains.backward()
        optimizer.step()

        metrics += loss_remains.item()
        y_pred.append(temp_outputs.to("cpu").detach().numpy().argmax(1))
        y_true.append(target.to("cpu").detach().numpy())
        
    metrics /= len(data)

    return metrics, np.hstack(y_pred), np.hstack(y_true)


def eval_fn(model, loss, data):
    model.eval()
    metrics = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(data, leave=False, desc="Eval"):
            tokens_list, target = batch
            target = target.to(device)
            temp_outputs = []
    
            for tokens in tokens_list:
                output = model(tokens)
                temp_outputs.append(output)
            temp_outputs = torch.stack(temp_outputs)
            
            loss_remains = loss(temp_outputs, target)

            metrics += loss_remains.item()
            y_pred.append(temp_outputs.to("cpu").detach().numpy().argmax(1))
            y_true.append(target.to("cpu").detach().numpy())
            
    metrics /= len(data)
    
    return metrics, np.hstack(y_pred), np.hstack(y_true)

def fit(model, loss, optimizer, scheduler, train_loader, valid_loader, device, epochs):
    info = "Epoch: %s Train loss: %.3f Valid loss: %.3f"
    best_macro = 0.0
    best_micro_precision_score = 0.0
    info_metrics = "Macro f1: %.3f | Macro precision: %.3f | Macro recall: %.3f | Micro precision: %.3f"
    
    for epoch in trange(epochs):
        train_loss, y_train_pred, y_train_true = train_fn(model, loss, train_loader, optimizer)
        eval_loss, y_eval_pred, y_eval_true = eval_fn(model, loss, valid_loader)
        
        train_metrics = get_score(y_train_true, y_train_pred)
        eval_metrics = get_score(y_eval_true, y_eval_pred)
        print(info_metrics % train_metrics , "-- train")
        print(info_metrics % eval_metrics, "-- eval")
        
        if best_macro < sum(eval_metrics[:3]) / 3:
            best_macro = sum(eval_metrics[:3]) / 3
            torch.save(model.state_dict(), f"./models/model_best_val_macro_{best_macro:.4f}.pt")

        if best_micro_precision_score < eval_metrics[-1]:
            best_micro_precision_score = eval_metrics[-1]
            torch.save(model.state_dict(), f"./models/model_best_val_micro_{best_micro_precision_score:.4f}.pt")
            
        
        print(info % (epoch + 1, train_loss, eval_loss), "\n")
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
    print(best_macro, best_micro_precision_score)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def collate_fn(data):
    data, label = list(zip(*data))
    return data, torch.stack(label)

text_columns = "vacancy_description"
target_columns = "target"

df = pd.read_csv("data_with_text.csv")
df["search_by_name"] = df["search_by_name"].apply(lambda x: x.split("_")[0]) # add next
# df[target_columns] = (df[target_columns] > 0).apply(lambda x: int(x))
df = df[df[target_columns] != 0]
print(df[target_columns].value_counts())

n_epoch = 5
info_metrics = "Macro f1: %.3f | Macro precision: %.3f | Macro recall: %.3f | Micro precision: %.3f"

train, test = train_test_split(df, shuffle=True, train_size=0.85, stratify=df[target_columns], random_state=random_state)

chunk_size = 1024

test_dataset = CustomVacancyDataset(test, tokenizer, device, text_columns, target_columns, chunk_size, mode="valid")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


if __name__ == "__main__":
    print(" Start train!")
    best_test_macro = 0.0
    best_test_micro_precision_score = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for train_index, val_index in skf.split(train[text_columns], train[target_columns]):
        bert_model = AutoModel.from_pretrained(model_name)
        model = BertClassification(bert_model).to(device)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr= 2e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset = CustomVacancyDataset(train.iloc[train_index], tokenizer, device, text_columns, target_columns, chunk_size, mode="train")
        valid_dataset = CustomVacancyDataset(train.iloc[val_index], tokenizer, device, text_columns, target_columns, chunk_size, mode="valid")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        fit(model, loss, optimizer, scheduler, train_loader, valid_loader, device, n_epoch)
        
        eval_loss, y_eval_pred, y_eval_true = eval_fn(model, loss, test_loader)
        eval_metrics = get_score(y_eval_true, y_eval_pred)
        print(info_metrics % eval_metrics, "-- test")

        if best_test_macro < sum(eval_metrics[:3]) / 3:
                best_macro = sum(eval_metrics[:3]) / 3
                torch.save(model.state_dict(), f"./models/model_best_test_macro_{best_macro:.4f}.pt")

        if best_test_micro_precision_score < eval_metrics[-1]:
            best_micro_precision_score = eval_metrics[-1]
            torch.save(model.state_dict(), f"./models/model_best_test_micro_{best_micro_precision_score:.4f}.pt")
