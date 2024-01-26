import os
import json
import jsonlines
from collections import defaultdict
from constant import spot_labels, spot_prompt, asoc_prompt
from utils import random
from utils import sequence_padding

 
class CLaREDataProcessor(object):
    def __init__(self, args, tokenizer):
        root = args.data_dir
        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length
        self.train_path = os.path.join(root, 'train.json')
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, '65_schemas.json')
        self.len_to_label = args.len_to_label
        self._load_schema()
        
    def get_train_sample(self):
        return self._pre_process(self.train_path)

    def get_dev_sample(self):
        with jsonlines.open(self.dev_path, 'r') as f:
            data_list = [line for line in f]
        return data_list

    def encoder(self, item):
        args = self.args
        text = item["text"]
        padding = 'max_length' if args.is_padding else False
        if args.model_type == 'roberta':
            encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, padding=padding, truncation=True)
            input_ids = encoder_text["input_ids"]
            token_type_ids = [0 for i in range(len(input_ids))]
            attention_mask = encoder_text["attention_mask"]
        elif args.model_type == 'albert' or args.model_type == 'bert':
            encoder_text = self.tokenizer(text, max_length=self.max_length, padding=padding, truncation=True, return_offsets_mapping=True)
            input_ids = encoder_text["input_ids"]
            token_type_ids = encoder_text["token_type_ids"]
            attention_mask = encoder_text["attention_mask"]
            offset_mapping = encoder_text['offset_mapping']
        spoes = set()
        if 'CMeIE' in args.data_dir:
            for spo_item in item["spo_list"]:
                sub = spo_item['subject']
                s_t = spo_item['subject_type']
                obj = spo_item['object']
                o_t = spo_item['object_type']
                p = spo_item['predicate']
                s = self.tokenizer.encode(sub, add_special_tokens=False) 
                p = self.schema.get(s_t + "_" + p + "_" +o_t,-1) if args.with_entity_type else self.schema.get(p,-1)
                o = self.tokenizer.encode(obj, add_special_tokens=False)

                if 'sub_char_span' not in spo_item.keys():
                    sh = self.search_chn(s, input_ids)
                    if sh == -1:
                        ent_start = text.find(sub)
                        if ent_start != -1:
                            ent_end = ent_start + len(sub) -1
                            spo_item['sub_char_span'] = [ent_start,ent_end]

                if 'sub_char_span' in spo_item.keys():
                    sh = self.search_with_char_span_chn(spo_item['sub_char_span'], offset_mapping)

                if 'obj_char_span' not in spo_item.keys():
                    oh = self.search_chn(o, input_ids)
                    if oh == -1:
                        ent_start = text.find(obj)
                        if ent_start != -1:
                            ent_end = ent_start + len(obj) -1
                            spo_item['obj_char_span'] = [ent_start,ent_end]

                if 'obj_char_span' in spo_item.keys():
                    oh = self.search_with_char_span_chn(spo_item['obj_char_span'], offset_mapping)
                calc_st = sh+len(s)-1
                calc_ot = oh+len(o)-1
                if calc_st >= args.max_length -1 or calc_ot >= args.max_length -1:
                    print('too long skip')
                    continue
                if sh != -1 and oh != -1:
                    if args.er_model_type == 'er-h':
                        spoes.add((sh, len(s), p, oh, len(o)))
                    elif args.er_model_type == 'er-t':
                        spoes.add((sh+len(s)-1, len(s), p, oh+len(o)-1, len(o)))
                    elif args.er_model_type == 'er-shot':
                        spoes.add((sh, len(s), p, oh+len(o)-1, len(o)))
                    elif args.er_model_type == 'er-stoh':
                        spoes.add((sh+len(s)-1, len(s), p, oh, len(o)))
                else:
                    print('\ndata err at item:{}'.format(item))
                    print(sub, p, obj, s_t, o_t,len(input_ids))
        else:
            for sub, p, obj, s_t, o_t in item["spo_list"]:
                p = self.schema.get(s_t + "_" + p + "_" +o_t,-1) if args.with_entity_type else self.schema.get(p,-1)
                s = self.tokenizer.encode(sub, add_special_tokens=False) 
                s_with_space = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                o = self.tokenizer.encode(obj, add_special_tokens=False)
                o_with_space = self.tokenizer.encode(' '+obj, add_special_tokens=False)
                sh = self.search_eng(s, s_with_space, input_ids)
                oh = self.search_eng(o, o_with_space, input_ids)
                if sh != -1 and oh != -1:
                    if args.er_model_type == 'er-h':
                        spoes.add((sh, len(s), p, oh, len(o)))
                    elif args.er_model_type == 'er-t':
                        spoes.add((sh+len(s)-1, len(s), p, oh+len(o)-1, len(o)))
                    elif args.er_model_type == 'er-shot':
                        spoes.add((sh, len(s), p, oh+len(o)-1, len(o)))
                    elif args.er_model_type == 'er-stoh':
                        spoes.add((sh+len(s)-1, len(s), p, oh, len(o)))
                else:
                    print('\ndata error at item:{}'.format(item))
                    print(sub, p, obj, s_t, o_t)
        # 根据对ace数据的统计，限制每个实体长度最长为6
        rel_labels = [set() for i in range(len(self.schema))]
        exist_pairs = list(args.len_to_label.keys())
        num_entity_labels = len(exist_pairs)
        entity_labels = [set() for i in range(num_entity_labels)]
        for s, len_s, p, o, len_o in spoes:
            if (len_s,len_o) not in exist_pairs:
                continue
            i = (len_s,len_o)
            if args.er_model_type == 'er-h':
                sh, oh, = s, o
                st = sh + len_s - 1
                ot = oh + len_o - 1
                entity_labels[self.len_to_label[i]].add((sh, oh))
            elif args.er_model_type == 'er-t':
                st, ot, = s, o
                sh = st - len_s + 1
                oh = ot - len_o + 1
                entity_labels[self.len_to_label[i]].add((st, ot))
            elif args.er_model_type == 'er-shot':
                sh, ot = s, o
                st = sh + len_s - 1
                oh = ot - len_o + 1
                entity_labels[self.len_to_label[i]].add((sh, ot))
            elif args.er_model_type == 'er-stoh':
                st, oh = s, o
                sh = st - len_s + 1
                ot = oh + len_o - 1
                entity_labels[self.len_to_label[i]].add((st, oh))
            if p != -1:
                # 修改这里，这一个矩阵都视为标签
                for s_index in range(sh,st+1):
                    for o_index in range(oh,ot+1):
                        rel_labels[p].add((s_index,o_index))
        for label in entity_labels+rel_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        rel_labels = sequence_padding([list(l) for l in rel_labels])
        return entity_labels, rel_labels, input_ids, token_type_ids, attention_mask



    def get_test_sample(self):
        with jsonlines.open(self.test_path, 'r') as f:
            data_list = [line for line in f]
        return data_list
    
    # ****************************英文****************************
    def search_eng(self, pattern, pattern_with_space, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern or sequence[i:i + n] == pattern_with_space:
                return i
        return -1
    # ****************************英文****************************

    # ****************************中文****************************
    def search_chn(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def search_with_char_span_chn(self, char_span, offset_mapping):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        for i, (start, end) in enumerate(offset_mapping):
            # start与end都是0，表示是special tokens
            if start==0 and end == 0:
                continue
            if start >= int(char_span[0]) and end <= int(char_span[1]) + 1:
                return i
        return -1
    # ****************************中文****************************
    
    def _load_schema(self):    
        labels  = set()
        predicates = set()
        with jsonlines.open(self.schema_path,'r') as f:
            for line in f:
                labels.add(line['subject_type']) 
                labels.add(line['object_type']) 
                predicates.add(line['predicate']) 
        labels = list(labels)
        predicates = list(predicates)
        labels = sorted(labels)
        predicates = sorted(predicates)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        self.labels = labels
        self.predicates = predicates
        self.num_predicates = len(predicates)
        
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        schema, id2predicate = {}, {}
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                if self.args.with_entity_type:
                    schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
                    id2predicate[idx] = item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]


        if not self.args.with_entity_type:
            for idx, item in enumerate(predicates):
                schema[item] = idx
                id2predicate[idx] = item
        print('schema:{}'.format(schema))
        print('id2predicate:{}'.format(id2predicate))
        self.id2predicate = id2predicate
        self.schema = schema

    def _pre_process(self, path):
        new_data = []
        if 'CMeIE' in self.args.data_dir:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    new_data.append({
                        "text":line["text"],
                        "spo_list":line["spo_list"]
                    })
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"], spo["subject_type"], spo["object_type"])
                                    for spo in line["spo_list"]]
                    })
        return new_data
  

