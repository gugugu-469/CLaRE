import torch
import torch.nn as nn
import os
import json
import jsonlines
import shutil
import math
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args, SPO, ACESPO
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from optimizer import GPLinkerOptimizer
import wandb
import time

class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
            test_dataset=None,
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        if test_dataset is not None and isinstance(test_dataset, Dataset):
            self.test_dataset = test_dataset

        self.logger = logger

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model

        self.output_dir = self.args.output_dir
        save_args(args, self.output_dir)

        
        if args.distributed == True:
            model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        else:
            model.to(args.device)
            
        
        
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)
        # skip_evaluate_models：妥协做法，以后应该把skip_evaluate单独设置在主函数的args中
        skip_evaluate_models = ['gplinker', 'gpner', 'gpner2', 'gpner4', 'gpner6', 'gpner7', 'gpner8', 'gpner9', 'ace05', 'gpfilter', 'biaffinefilter', 'gpner11', 'bibm_dual', 'gpner14', 'gpner15', 'gpnersub', 'gpnerobj', 'gpner16',  'gplinkerace05']
        
        if args.method_name in skip_evaluate_models:
            args.skip_evaluate = True
        else:
            args.skip_evaluate = False
        if args.optimizer == 'Adam':
            optimizer = GPLinkerOptimizer(model, args, warmup_proportion=args.warmup_proportion,num_training_steps=num_training_steps)
        elif args.optimizer == 'AdamW':
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            raise ValueError('Optimizer值指定错误:{}'.format(args.optimizer))
        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_epoch = None
        best_score = 0
        cnt_patience = 0

        ####
        best_test_f = 0
        best_test_index = -1
        ####
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'val_p', 'val_r', 'val_f1'])
        # 统计指标
        metric = d2l.Accumulator(5)
        num_batches = len(train_dataloader)
        
        
        for epoch in range(args.epochs):
            train_start = time.time()
            print('**********Now Epoch: {} **********'.format(epoch))
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.optimizer != 'Adam':
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    animator.add(
                            global_step / num_batches, 
                            (loss / args.loss_show_rate, train_p, train_r, train_f1, 0, 0, 0))
                    d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

            train_end = time.time()
            print('训练一个EPOCH时间:{}'.format(train_end-train_start))
            if not args.skip_evaluate and epoch >= 5:
                val_p, val_r, val_f1 = self.evaluate(model)
                animator.add(
                    global_step / num_batches, 
                    (# metric[0] / metric[-1] / args.loss_show_rate, # loss太大，除以loss_show_rate才能在[0,1]范围内看到
                     loss / args.loss_show_rate,
                     train_p,  # metric[1] / metric[-1],
                     train_r,  # metric[2] / metric[-1],
                     train_f1, # metric[3] / metric[-1],
                     val_p,
                     val_r,
                     val_f1))
                d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

                if args.save_metric == 'step':
                    save_metric = global_step
                elif args.save_metric == 'epoch':
                    save_metric = epoch
                elif args.save_metric == 'loss':
                    # e的700次方刚好大于0，不存在数值问题
                    # 除以10，避免loss太大，exp(-loss)次方由于数值问题会小于0，导致存不上，最大可以处理7000的loss
                    save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
                elif args.save_metric == 'p':
                    save_metric = val_p
                elif args.save_metric == 'r':
                    save_metric = val_r
                elif args.save_metric == 'f1':
                    save_metric = val_f1
                if save_metric > best_score:
                    best_score = save_metric
                    best_step = global_step
                    best_epoch = epoch
                    cnt_patience = 0
                    self.args.loss = loss # metric[0] / metric[-1]
                    self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                                     #  metric[1] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                    self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                    self._save_checkpoint(model)
                else:
                    cnt_patience += 1
                    print('Early Stop:{} out of {}'.format(cnt_patience,args.earlystop_patience))
            else:
                print('SKIP EVALUATE')

            if cnt_patience >= args.earlystop_patience:
                print('*****Early Stopped! Training Over!*****')
                break

            if args.skip_evaluate:
                self.args.loss = loss
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                self._save_checkpoint(model)
        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}, best epoch: {best_epoch}\n" )

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model,error=False):
        args = self.args
        
        if args.distributed:
            model=model.module
        if not error:
            model = model.to(torch.device('cpu'))
            torch.save(model.state_dict(), os.path.join(self.output_dir, 'pytorch_model.pt'))
            self.logger.info('Saving models checkpoint to %s', self.output_dir)
            self.tokenizer.save_vocabulary(save_directory=self.output_dir)
            model = model.to(args.device)
            save_args(args, self.output_dir)
            shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                            os.path.join(self.output_dir, 'config.json'))
        else:
            save_path = os.path.join(self.output_dir,'error_model')
            os.makedirs(save_path,exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'pytorch_model.pt'))
            self.logger.info('Saving models checkpoint to %s', save_path)
            self.tokenizer.save_vocabulary(save_directory=save_path)
            model = model.to(args.device)
            save_args(args, save_path)
            shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                            os.path.join(save_path, 'config.json'))
    
    
    def load_checkpoint(self):
        args = self.args
        load_dir = os.path.join(args.output_dir)
        self.logger.info(f'load model from {load_dir}')
        # 每次加载到cpu中，防止爆显存
        checkpoint = torch.load(os.path.join(load_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn_test if hasattr(self.eval_dataset, 'collate_fn_test') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, batch_size=None):
        collate_fn = self.test_dataset.collate_fn_test if hasattr(self.test_dataset, 'collate_fn_test') else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )


        
class CLaRETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None,
            test_dataset=None
    ):
        super(CLaRETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            logger=logger,
        )
        self.best_pair_f1 = 0.
        self.best_trip_f1 = 0.
        self.label_to_len = args.label_to_len
        self.sigmoid = torch.nn.Sigmoid()

    def training_step(self, model, item):
        args = self.args
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_token_type_ids, batch_mask_ids, batch_entity_labels, batch_rel_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_rel_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1,  loss2]) / 2
        loss.backward()

        p = 0
        r = 0
        f1 = 0
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        return self.predict(mode='dev')
        
    def get_pair_metric(self,mode='test'):
        args = self.args
        golds = []
        preds = []
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
            output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version,'entity_pair_test_{}.jsonl'.format(args.entity_threshold))
        elif mode=='dev':
            test_samples = self.data_processor.get_dev_sample()
            output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version,'entity_pair_dev_{}.jsonl'.format(args.entity_threshold))
        else:
            test_samples = self.data_processor.get_dev_sample()
            output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version,'entity_pair_test_dev_{}.jsonl'.format(args.entity_threshold))

        for index,data in enumerate(test_samples):
            data = data['spo_list']
            for ent in data:
                golds.append((index,ent['subject'].strip(),ent['object'].strip()))
        with jsonlines.open(output_dir,'r') as f:
            for index,data in enumerate(f):
                data = data['spo_list']
                for ent in data:
                    preds.append((index,ent['subject'].strip(),ent['object'].strip()))
        try:
            P = len(set(preds) & set(golds)) / len(set(preds))
            R = len(set(preds) & set(golds)) / len(set(golds))
            F = (2 * P * R) / (P + R)
        except:
            P = 0
            R = 0
            F = 0

        return P,R,F

    def cal_prf1(self, mode='dev'):
        args = self.args
        output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version)
        if mode == 'test':
            predict_dir = os.path.join(output_dir, 'test_{}_{}.jsonl'.format(args.entity_threshold,args.re_threshold))
            file_name = 'test.json'
        elif mode=='dev':
            predict_dir = os.path.join(output_dir, 'dev_{}_{}.jsonl'.format(args.entity_threshold,args.re_threshold))
            file_name = 'dev.json'
        else:
            predict_dir = os.path.join(output_dir, 'test_dev_{}_{}.jsonl'.format(args.entity_threshold,args.re_threshold))
            file_name = 'dev.json'
        print('predict_path:{}'.format(predict_dir))

        with jsonlines.open(predict_dir, mode='r') as r:
            pred_datas = [i for i in r]
        with jsonlines.open(os.path.join(args.data_dir,file_name), mode='r') as r:
            gold_datas = [i for i in r]
        preds = []
        golds = []
        for index,data in enumerate(pred_datas):
            for spo in data['spo_list']:
                if args.with_entity_type:
                    preds.append((index,spo['subject'],spo['subject_type'],spo['predicate'],spo['object'],spo['object_type']))
                else:
                    preds.append((index,spo['subject'],'entity',spo['predicate'],spo['object'],'entity'))

        for index,data in enumerate(gold_datas):
            for spo in data['spo_list']:
                if args.with_entity_type:
                    golds.append((index,spo['subject'],spo['subject_type'],spo['predicate'],spo['object'],spo['object_type']))
                else:
                    golds.append((index,spo['subject'],'entity',spo['predicate'],spo['object'],'entity'))

        try:
            P = len(set(preds) & set(golds)) / len(set(preds))
            R = len(set(preds) & set(golds)) / len(set(golds))
            F = (2 * P * R) / (P + R)
        except:
            P = 0
            R = 0
            F = 0
        return P,R,F
    

    def predict(self,mode='test'):
        args = self.args
        logger = self.logger
        model = self.model
        device = args.device

        if mode == 'test':
            test_dataloader = self.get_test_dataloader()
        elif mode == 'test_dev':
            test_dataloader = self.get_eval_dataloader()
        elif mode == 'dev':
            test_dataloader = self.get_eval_dataloader()
        num_examples = len(test_dataloader.dataset)


        id2predicate = self.data_processor.id2predicate
        model.to(device)
        print_text = 'Predicting' if mode == 'test' else 'Evaluating'
        logger.info(f"***** Running {print_text} *****")
        print('PREDICT_ENTITY_THRESHOLD:{}'.format(args.entity_threshold))
        logger.info("Num samples %d", num_examples)
        
        pred_data_sp = []
        pred_data = []
        pbar = ProgressBar(n_total=len(test_dataloader), desc=print_text)
        padding = 'max_length' if args.is_padding else False
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for step, item in enumerate(test_dataloader):
                pbar(step)
                texts = item
                encoder_text = self.tokenizer(texts, return_offsets_mapping=True, max_length=self.args.max_length, truncation=True,padding=True,return_tensors='pt')
                input_ids = encoder_text['input_ids']
                attention_mask = encoder_text['attention_mask']
                token2char_span_mapping = encoder_text['offset_mapping']
                valid_length = (torch.sum(attention_mask,dim=-1)-1).tolist()
                text_num = len(input_ids)
                valid_length = [[0]*text_num,valid_length]
                valid_index = list(range(text_num))
                multi_mask = attention_mask.clone()
                multi_mask[valid_index,valid_length] = 0
                multi_mask_1 = multi_mask.unsqueeze(dim=-1).unsqueeze(dim=1).to(device)
                multi_mask_2 = multi_mask.unsqueeze(dim=1).unsqueeze(dim=1).to(device)


                # 拿到输出
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                scores = model(input_ids, attention_mask)
                pair_outputs = self.sigmoid(scores[0])
                pair_outputs = (pair_outputs*multi_mask_1*multi_mask_2).data.cpu().numpy()
                # print('pair_outputs:{}'.format(pair_outputs))
                re_outputs = scores[1]

                sp_pairs = set()

                spoes_ent_pair = defaultdict(list)
                spoes = defaultdict(list)

                for text_index, l, s, o in zip(*np.where(pair_outputs > args.entity_threshold)):
                    text = texts[text_index]
                    # 从label映射到len
                    s_len,o_len = self.label_to_len[l]
                    if args.er_model_type == 'er-h':
                        sh, oh = s, o
                        st = sh + s_len - 1
                        ot = oh + o_len - 1
                    elif args.er_model_type == 'er-t':
                        st, ot = s, o
                        sh = st - s_len + 1
                        oh = ot - o_len + 1
                    elif args.er_model_type == 'er-shot':
                        sh, ot = s, o
                        st = sh + s_len - 1
                        oh = ot - o_len + 1
                    elif args.er_model_type == 'er-stoh':
                        st, oh = s, o
                        sh = st - s_len + 1
                        ot = oh + o_len - 1
                    text_token2char_span_mapping = token2char_span_mapping[text_index]
                    if sh<0 or st >= len(text_token2char_span_mapping) or oh < 0 or ot >= len(text_token2char_span_mapping):
                        continue
                    if text_token2char_span_mapping[sh][0] == text_token2char_span_mapping[sh][1] or text_token2char_span_mapping[st][0] == text_token2char_span_mapping[st][1] or text_token2char_span_mapping[oh][0] == text_token2char_span_mapping[oh][1] or text_token2char_span_mapping[ot][0] == text_token2char_span_mapping[ot][1]:
                        continue
                    sp_pairs.add((text_index, sh, st, oh, ot))
                sp_pairs = list(sp_pairs)
                # 输出中间结果(主客体对)
                
                max_len = re_outputs.shape[-1]
                for text_index, sh, st, oh, ot in sp_pairs:
                    text = texts[text_index]
                    text_token2char_span_mapping = token2char_span_mapping[text_index]
                    ###### 处理实体对
                    sub_name = text[text_token2char_span_mapping[sh][0]:text_token2char_span_mapping[st][-1]].strip()
                    obj_name = text[text_token2char_span_mapping[oh][0]:text_token2char_span_mapping[ot][-1]].strip()

                    spoes_ent_pair[text_index].append((
                        sub_name,
                        obj_name
                    ))
                    ##### 处理五元组
                    select_preds = re_outputs[text_index,:,sh:st+1, oh:ot+1]
                    select_preds_mean = torch.mean(select_preds, dim=(-1, -2)).data.cpu().numpy()
                    ps = np.where(select_preds_mean > args.re_threshold)[0]
                    for p in ps:
                        spoes[text_index].append((
                            sub_name, id2predicate[p], obj_name
                        ))
            
                # 实体对
                for text_index in range(text_num):
                    tmp_spoes_ent_pair = spoes_ent_pair[text_index]
                    tmp_spoes = spoes[text_index]
                    text = texts[text_index]
                    sp_list = []
                    for spo in tmp_spoes_ent_pair:
                        sp_list.append({"subject":spo[0],"object":spo[1]})
                    pred_data_sp.append({"text":text, "spo_list":sp_list})
                    # 五元组
                    spo_list = []
                    for spo in list(tmp_spoes):
                        if args.model_type == 'albert':
                            spo = list(spo)
                            if spo[0] and spo[0][0] == ' ':
                                spo[0] = spo[0][1:]
                            if spo[2] and spo[2][0] == ' ':
                                spo[2] = spo[2][1:]
                        if args.with_entity_type:
                            spo_list.append({"predicate":'_'.join(spo[1].split("_")[1:-1]), "subject":spo[0], "subject_type":spo[1].split("_")[0],
                                            "object":spo[2], "object_type": spo[1].split("_")[-1]})
                        else:
                            spo_list.append({"predicate":spo[1], "subject":spo[0], "object":spo[2]})
                    pred_data.append({"text":text, "spo_list":spo_list})
            if mode == 'test':
                output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir,exist_ok=True)
                # 输出SP pairs
                output_dir_sp = os.path.join(output_dir, 'entity_pair_test_{}.jsonl'.format(args.entity_threshold))
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_sp, mode='w') as f:
                    for data in pred_data_sp:
                        f.write(data)
                # 输出五元组
                output_dir_trips = os.path.join(output_dir, 'test_{}_{}.jsonl').format(args.entity_threshold,args.re_threshold)
                
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_trips, mode='w') as f:
                    for data in pred_data:
                        f.write(data)
                print('\nentity threshold:{}'.format(args.entity_threshold))
                p,r,f = self.get_pair_metric(mode)
                print('TEST ENTITY PAIR Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                p,r,f = self.cal_prf1(mode)
                print('TEST TRIP Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                prf_time = time.time()
                print('prf use time:{}'.format(prf_time-start_time))
                return p,r,f
            elif mode == 'dev':
                output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir,exist_ok=True)
                # 输出SP pairs
                output_dir_sp = os.path.join(output_dir, 'entity_pair_dev_{}.jsonl'.format(args.entity_threshold))
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_sp, mode='w') as f:
                    for data in pred_data_sp:
                        f.write(data)
                # 输出五元组
                output_dir_trips = os.path.join(output_dir, 'dev_{}_{}.jsonl'.format(args.entity_threshold,args.re_threshold))
                
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_trips, mode='w') as f:
                    for data in pred_data:
                        f.write(data)
                print('EVAL ON DEV')
                p,r,f = self.get_pair_metric(mode)
                if f > self.best_pair_f1:
                    print('FIND BEST PAIR F1')
                    self.best_pair_f1 = f
                print('DEV ENTITY PAIR Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                p,r,f = self.cal_prf1(mode)
                print('DEV TRIP Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                if f > self.best_trip_f1:
                    print('FIND BEST TRIP F1')
                    self.best_trip_f1 = f
                return p,r,f
            else:
                output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name,args.model_version)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir,exist_ok=True)
                # 输出SP pairs
                output_dir_sp = os.path.join(output_dir, 'entity_pair_test_dev_{}.jsonl'.format(args.entity_threshold))
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_sp, mode='w') as f:
                    for data in pred_data_sp:
                        f.write(data)
                # 输出五元组
                output_dir_trips = os.path.join(output_dir, 'test_dev_{}_{}.jsonl'.format(args.entity_threshold,args.re_threshold))
                logger.info(f"***** write predict file to {output_dir} *****")
                with jsonlines.open(output_dir_trips, mode='w') as f:
                    for data in pred_data:
                        f.write(data)
                print('\nentity threshold:{}'.format(args.entity_threshold))
                p,r,f = self.get_pair_metric(mode)
                print('TEST DEV ENTITY PAIR Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                p,r,f = self.cal_prf1(mode)
                print('TEST DEV TRIP Metrics: P:{}\tR:{}\tF:{}'.format(p,r,f))
                return p,r,f           
        
