#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
# from classification_sup import load_dataset
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel
from transformers import EarlyStoppingCallback
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class ClassificationModel(nn.Module):
    def __init__(self, encoder_path, base_model_name="FacebookAI/xlm-roberta-base", hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        # self.encoder.load_state_dict(torch.load(encoder_path), strict=False)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

class TextPairDataset(Dataset):
    def __init__(self, tokenizer, text1, text2, labels, max_length=512):
        self.tokenizer = tokenizer
        self.text1 = text1
        self.text2 = text2
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = str(self.text1[idx])
        text2 = str(self.text2[idx])
        label = int(self.labels[idx])

        inputs = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# === Dataset loader ===
def load_dataset(df, tokenizer, max_length=512):
    return TextPairDataset(
        tokenizer=tokenizer,
        text1=df["premise"].tolist(),
        text2=df["hypothesis"].tolist(),
        labels=df["labels"].tolist(),
        max_length=max_length
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    f1_pos = f1_score(labels, preds, pos_label=1, average='binary')  # F1 dla klasy 1
    acc = np.mean(preds == labels)

    return {
        'accuracy': acc,
        'f1_pos': f1_pos
    }


# ===== Model z trenowania =====
class ClassificationModel(nn.Module):
    def __init__(self, encoder_path, hidden_size=768):
        super().__init__()
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("FacebookAI/xlm-roberta-base")
        self.encoder = AutoModel.from_config(config)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


# ===== Pomocnicze =====
def load_paragraphs_from_file(path: str) -> list[str]:
    with open(path, encoding="utf8") as file:
        content = file.read()
    return content.split('\n')


# def make_pairs(para_list):
#     return [{"text1": para_list[i], "text2": para_list[i+1], "label": 0} for i in range(len(para_list) - 1)]

def make_pairs(para_list):
    return [{"premise": para_list[i], "hypothesis": para_list[i+1], "labels": 0} for i in range(len(para_list) - 1)]


def predict_pairs(model, dataset, device):
    preds = []
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=16)
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().tolist())
    return preds


# ===== Główna logika =====
def main(args):
    prediction_models = {
        'easy':   './models/trained_on_contrastive_encoder_10_epoch_capslock_easy_freeze_0',
        'medium': './models/trained_on_contrastive_encoder_10_epoch_question_medium_freeze_0',
        'hard':   './models/trained_on_contrastive_encoder_10_epoch_capslock_hard_freeze_0',
    }

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    for difficulty in ['easy', 'medium', 'hard']:
        input_dir = os.path.join(args.input, difficulty)
        output_dir = os.path.join(args.output, difficulty)
        os.makedirs(output_dir, exist_ok=True)

        files_list = glob.glob(f'{input_dir}/**/*.txt', recursive=True)
        print(f'[{difficulty}] Znaleziono {len(files_list)} plików.')

        model_path = prediction_models[difficulty]
        model = ClassificationModel(encoder_path=model_path)

        # Wczytaj wagi
        model_safe_path = os.path.join(model_path, "model.safetensors")
        model_bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(model_safe_path):
            state_dict = load_file(model_safe_path)
        elif os.path.isfile(model_bin_path):
            state_dict = torch.load(model_bin_path, map_location=args.device)
        else:
            print(f"Brak wag w {model_path}, pomijam...")
            continue

        model.load_state_dict(state_dict, strict=False)
        model.to(args.device)

        for document_path in tqdm(files_list):
            share_id = os.path.basename(document_path)[8:-4]
            para_list = load_paragraphs_from_file(document_path)

            # Tworzenie par i datasetu
            # pair_data = make_pairs(para_list)
            # test_dataset = load_dataset(pair_data, tokenizer)
            import pandas as pd

            # pair_data = make_pairs(para_list)
            # pair_df = pd.DataFrame(pair_data)
            # pair_df.rename(columns={"text1": "premise", "text2": "hypothesis"}, inplace=True)

            # test_dataset = load_dataset(pair_df, tokenizer)
            import pandas as pd

            pair_data = make_pairs(para_list)
            pair_df = pd.DataFrame(pair_data)

            test_dataset = load_dataset(pair_df, tokenizer)



            # Predykcja
            prediction = predict_pairs(model, test_dataset, args.device)
            result = {"changes": prediction}

            output_file = os.path.join(output_dir, f"solution-problem-{share_id}.json")
            with open(output_file, 'w', encoding='utf8') as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="../pan24_dataset/test", type=str)
    parser.add_argument("-o", "--output", default="../pan24_dataset/test", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-e", "--enrich", action='store_true')  # dla zgodności
    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    main(args)
