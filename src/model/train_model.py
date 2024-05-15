import os
from dotenv import load_dotenv
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import torch
from torch import nn
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from model.model_utils import get_entities
from model.EntityExtractionBERT import EntityExtractionBERT
from src.model.train_utils import get_metrics_dict

load_dotenv()


class Trainer:
    def __init__(self, train_dataloader, dev_dataloader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataloader, self.dev_dataloader = train_dataloader, dev_dataloader
        entities = get_entities()
        self.entity_label_map = {i: label for i, label in enumerate(entities)}
        self.num_entities = len(entities)
        self.config = BertConfig.from_pretrained(os.environ.get('BERT_PRETRAINED'))
        self.model = EntityExtractionBERT(config=self.config)
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss(ignore_index=int(os.environ.get('PAD_LABEL_ID')))
        self.optimizer = self.__get_optimizer()
        self.scheduler = self.__get_scheduler()
        self.global_step = 0
        self.tr_loss = 0.0

    def __get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': float(os.environ.get('WEIGHT_DECAY'))},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(os.environ.get('LEARNING_RATE')),
                          eps=float(os.environ.get('ADAM_EPSILON')))

        return optimizer

    def __get_scheduler(self):
        t_total = len(self.train_dataloader) // int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) * int(
            os.environ.get('NUM_TRAIN_EPOCHS'))
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=float(os.environ.get('WARMUP_STEPS')),
                                                    num_training_steps=t_total)
        return scheduler

    def __get_loss(self, output, attention_mask, entity_labels_ids):
        entity_logits = output[0]
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = entity_logits.view(-1, self.num_entities)[active_loss]
            active_labels = entity_labels_ids.view(-1)[active_loss]
            slot_loss = self.loss(active_logits, active_labels)
        else:
            slot_loss = self.loss(entity_logits.view(-1, self.num_entities), entity_labels_ids.view(-1))
        return slot_loss

    def train(self):
        self.model.zero_grad()
        train_iterator = trange(int(os.environ.get('NUM_TRAIN_EPOCHS')), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            train_batch_loss = []
            valid_batch_loss = []
            valid_batch_metrics = get_metrics_dict()
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                loss = self.__get_loss(outputs, inputs['attention_mask'], batch[3])
                if int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) > 1:
                    loss = loss / int(os.environ.get('GRADIENT_ACCUMULATION_STEPS'))

                loss.backward()
                self.tr_loss += loss.item()

                if (step + 1) % int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(os.environ.get('MAX_GRAD_NORM')))

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    if int(os.environ.get('LOGGING_STEPS')) > 0 and self.global_step % int(
                            os.environ.get('LOGGING_STEPS')) == 0:
                        results, valid_loss = self.evaluate()
                        train_batch_loss.append(loss.item())
                        valid_batch_loss.append(valid_loss)
                        for key, value in results.items():
                            if key == 'accuracy':
                                valid_batch_metrics[key].append(value)
                            else:
                                for inner_key, inner_value in value.items():
                                    valid_batch_metrics[key][inner_key].append(inner_value)
                        print(f'train loss функция: {train_batch_loss[-1]}')
                        print(f'valid loss функция: {valid_batch_loss[-1]}')
                        print(f"macro avf f1: {results['macro avg']['f1-score']}")

                    if int(os.environ.get('SAVE_STEPS')) > 0 and self.global_step % int(
                            os.environ.get('SAVE_STEPS')) == 0:
                        self.save_model()

    def evaluate(self):
        entity_preds = None
        entity_labels_ids = None

        batch_loss = []
        self.model.eval()
        for batch in tqdm(self.dev_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                loss = self.__get_loss(outputs, inputs['attention_mask'], batch[3])
                batch_loss.append(loss.item())
                if entity_preds is None:
                    entity_preds = outputs[0].detach().cpu().numpy()
                    entity_labels_ids = batch[3].detach().cpu().numpy()
                else:
                    entity_preds = np.append(entity_preds, outputs[0].detach().cpu().numpy(), axis=0)
                    entity_labels_ids = np.append(entity_labels_ids,
                                                  batch[3].detach().cpu().numpy(),
                                                  axis=0)
        entity_preds = np.argmax(entity_preds, axis=2)
        entity_label_list = [[] for _ in range(entity_labels_ids.shape[0])]
        entity_preds_list = [[] for _ in range(entity_labels_ids.shape[0])]

        for i in range(entity_labels_ids.shape[0]):
            for j in range(entity_labels_ids.shape[1]):
                if entity_labels_ids[i, j] != int(os.environ.get('PAD_LABEL_ID')):
                    entity_label_list[i].append(self.entity_label_map[entity_labels_ids[i][j]])
                    entity_preds_list[i].append(self.entity_label_map[entity_preds[i][j]])

        entity_label_list = [label for sublist in entity_label_list for label in sublist]
        entity_preds_list = [label for sublist in entity_preds_list for label in sublist]
        report = classification_report(entity_label_list, entity_preds_list, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print(report_df)

        return report, np.mean(batch_loss)

    def save_model(self):
        if not os.path.exists(os.environ.get('SAVE_MODEL_DIR')):
            os.makedirs(os.environ.get('SAVE_MODEL_DIR'))
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(os.environ.get('SAVE_MODEL_DIR'))
