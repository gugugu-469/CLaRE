import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./codes')
import os
import argparse
import torch
import shutil
import random
from models import CLaREModel
from trainer import CLaRETrainer
from data import CLaREDataset, CLaREDataProcessor
from utils import init_logger, seed_everything, get_devices, get_time, zhank
import jsonlines
from collections import defaultdict
import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l
from transformers import BertTokenizerFast, AlbertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, RobertaModel, AlbertModel
MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (AutoTokenizer, RobertaModel),
    'mcbert': (AutoTokenizer, AutoModelForMaskedLM),
    'albert': (AutoTokenizer, AlbertModel)
}



def sort_count_value(count):
    return {k:count[k] for k in sorted(count.keys(),key = lambda x:count[x],reverse=True)}
def sort_count(count):
    return {k:count[k] for k in sorted(count.keys(),key = lambda x:x[0]*100+x[1])}


def get_label_list(args):
    print('开始获取label list')
    print('args.pair_filter:',args.pair_filter)
    print('args.use_random:',args.use_random)
    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=True)
    all_count = defaultdict(int)

    with jsonlines.open(os.path.join(args.data_dir,'train.json')) as f:
        for line in f:
            for spo in line['spo_list']:
                sub_length = len(tokenizer(spo['subject'],add_special_tokens=False)['input_ids'])
                obj_length = len(tokenizer(spo['object'],add_special_tokens=False)['input_ids'])
                all_count[(sub_length,obj_length)] += 1

    with jsonlines.open(os.path.join(args.data_dir,'dev.json')) as f:
        for line in f:
            for spo in line['spo_list']:
                sub_length = len(tokenizer(spo['subject'],add_special_tokens=False)['input_ids'])
                obj_length = len(tokenizer(spo['object'],add_special_tokens=False)['input_ids'])
                all_count[(sub_length,obj_length)] += 1
    test_count = defaultdict(int)
    with jsonlines.open(os.path.join(args.data_dir,'test.json')) as f:
        for line in f:
            for spo in line['spo_list']:
                sub_length = len(tokenizer(spo['subject'],add_special_tokens=False)['input_ids'])
                obj_length = len(tokenizer(spo['object'],add_special_tokens=False)['input_ids'])
                all_count[(sub_length,obj_length)] += 1
                test_count[(sub_length,obj_length)] += 1
    all_count = sort_count_value(all_count)
    test_count = sort_count_value(test_count)
    print('all_count:{}'.format(all_count))
    print('test_count:{}'.format(test_count))
    # 删除的key数
    delete_keys = 0
    # 关系总数
    all_nums = 0
    # 删除的关系数
    delete_values = 0
    for k,v in test_count.items():
        all_nums += v
        if v <= args.pair_filter:
            delete_keys += 1
            delete_values += v
    print('删除的key:{}\t删除的关系数:{}\t总关系数:{}\t占比:{}'.format(delete_keys,delete_values,all_nums,delete_values/all_nums))
    count_new = {k:v for k,v in all_count.items() if v > args.pair_filter}
    dict_keys = list(count_new.keys())
    print('dict_keys:{}'.format(dict_keys))
    if args.use_random:
        random.shuffle(dict_keys)
        print('dict_keys after shuffle:{}'.format(dict_keys))
    len_to_label = {dict_keys[index]:index for index in range(len(dict_keys))}
    return len_to_label



def get_args():
    parser = argparse.ArgumentParser()
    
    # 方法名：baseline required=True
    parser.add_argument("--method_name", default='CLaRE', type=str,
                        help="The name of method.")
    
    # 数据集存放位置：./CMeIE required=True
    parser.add_argument("--data_dir", default='./ACE05', type=str,
                        help="The task data directory.")
    
    # 预训练模型存放位置: /root/nas/Models required=True
    parser.add_argument("--model_dir", default='/root/nas/Models', type=str,
                        help="The directory of pretrained models")

    # 模型类型: bert required=True
    parser.add_argument("--model_type", default='bert', type=str, 
                        help="The type of selected pretrained models.")

    # 预训练模型: bert-base-chinese required=True
    parser.add_argument("--pretrained_model_name", default='RoBERTa_zh_Large_PyTorch', type=str,
                        help="The path or name of selected pretrained models.")
    
    # 微调模型: er required=True
    parser.add_argument("--finetuned_model_name", default='CLaRE', type=str,
                        help="The name of finetuned model")
    
    # 微调模型参数存放位置：./checkpoint required=True
    parser.add_argument("--output_dir", default='./checkpoint', type=str,
                        help="The path of result data and models to be saved.")
    
    # 是否训练：True
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    
    # 是否预测：False required=True
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the models in inference mode on the test set.")

    # 预测时加载的模型版本，如果做预测，该参数是必需的
    parser.add_argument("--model_version", default='', type=str,
                        help="model's version when do predict")
    
    # 提交结果保存目录：./result_output required=True
    parser.add_argument("--result_output_dir", default='./result_output', type=str,
                        help="the directory of commit result to be saved")
    
    # 设备：-1：CPU， i：cuda:i(i>0), i可以取多个，以逗号分隔 required=True
    parser.add_argument("--devices", default='0', type=str,
                        help="the directory of commit result to be saved")
    
    parser.add_argument("--loss_show_rate", default=200, type=int,
                        help="liminate loss to [0,1] where show on the train graph")
      
    # 序列最大长度：128
    parser.add_argument("--max_length", default=150, type=int,
                        help="the max length of sentence.")
    
    # 训练batch_size：32
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    
    # 评估batch_size：64
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    
    # 学习率：3e-5
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    # 权重衰退：取默认值
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    
    # 极小值：取默认值
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    
    # epochs：7
    parser.add_argument("--epochs", default=7, type=int,
                        help="Total number of training epochs to perform.")
    
    # 线性学习率比例：0.1
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    
    # earlystop_patience：100 （earlystop_patience step 没有超过最高精度则停止训练）
    parser.add_argument("--earlystop_patience", default=100, type=int,
                        help="The patience of early stop")
    
    # 多少step后打印一次：200
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")
    
    # 随机数种子：2021
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed for initialization")
    
    # 训练时保存 save_metric 最大存取模型 required=True
    parser.add_argument("--save_metric", default='f1', type=str,
                        help="the metric determine which model to save.")
    
    # 正则化手段，dropout
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="dropout rate")
    
    # GlobalPointer中的隐藏层维度
    parser.add_argument('--inner_dim', type=int, default=128,
                        help="inner dim of GlobalPointer")
    
    # 是否需要预测实体类型
    parser.add_argument('--with_entity_type', action='store_true',
                        help="If True, train and predict with entity type.")

    # NER模型的标注位置(主客体)
    parser.add_argument('--er_model_type', type=str, default='er-h',choices = ['er-h','er-t','er-shot','er-stoh'],
                        help="The model type of ner model. It is the label position of subject and object entity")
    
    # 是否需要补齐
    parser.add_argument('--is_padding', action='store_true',
                        help="whether to do padding")

    # 关系抽取采用的阈值
    parser.add_argument('--re_threshold', type=float, default=0,
                        help="The threshold of relation extraction when predicting.")

    # 优化器
    parser.add_argument('--optimizer', type=str, default='Adam', choices = ['AdamW','Adam'],
                        help="The optimizer of model.")

    # 实体对抽取采用的阈值
    parser.add_argument('--entity_threshold', type=float, default=0.5,
                        help="The threshold of entity pair extraction when predicting.")

    # 梯度裁剪
    parser.add_argument('--max_grad_norm', type=float, default=1.,
                        help="Used in training for gradient clipping.")
    # 是否在标签列表进行随机
    parser.add_argument('--use_random', action='store_true',
                        help="Whether to shuffle in label list.")

    # pair对出现次数的过滤
    parser.add_argument('--pair_filter', type=int, default=5,
                        help="Filter out only a small number of entity pair labels.")


    args = parser.parse_args()
    # 判断参数规范性
    args.time = get_time('%m-%d-%H-%M-%S')
    if args.do_predict and args.model_version == '':
        raise Exception('做预测的话必须提供加载的模型版本')    
    if args.do_train and args.model_version == '':
        args.model_version = args.time
        print('训练模型，未指定模型名称，自动以运行时间代替:{}'.format(args.model_version))
    # 判断是否多卡
    args.devices = get_devices(args.devices.split(','))
    args.device = args.devices[0]
    args.distributed = True if len(args.devices) > 1  else False 
    # 创建保存文件夹以及输出文件夹
    args.finetuned_model_name += '_' + args.er_model_type
    args.output_dir = os.path.join(args.output_dir, args.method_name,args.pretrained_model_name,args.finetuned_model_name,args.model_version)
    print('模型所有文件将保存至:{}'.format(args.output_dir))
    os.makedirs(args.output_dir,exist_ok=True)
    args.result_output_dir = os.path.join(args.result_output_dir, args.method_name) 
    os.makedirs(args.result_output_dir,exist_ok=True)
    print('模型预测输出文件目录:{}'.format(args.result_output_dir))

    print('ori seed:{}'.format(args.seed))
    if args.seed == -1:
        args.seed = random.randint(0,10000)
    print('final args.seed:{}'.format(args.seed))
    seed_everything(args.seed)
    if args.do_train:
        # 先加载label_list
        args.len_to_label = get_label_list(args)
        # 保存一下label_lists
        torch.save(args.len_to_label, os.path.join(args.output_dir, 'label_lists.pt'))
    elif args.do_predict:
        print('试图从此位置加载label_lists用于预测:{}'.format(os.path.join(args.output_dir, 'label_lists.pt')))
        args.len_to_label = torch.load(os.path.join(args.output_dir, 'label_lists.pt'))
    args.label_to_len = {v:k for k,v in args.len_to_label.items()}
    print('得到的label_to_len为:{}'.format(args.label_to_len))
    print('得到的len_to_label为:{}'.format(args.len_to_label))

    return args

def main():
    args = get_args()
    if 'ade' in args.data_dir.lower() and not args.use_macro:
        print('请注意，使用ADE数据:{},但是没有使用macro'.format(args.data_dir))
    logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    if args.do_train:
        logger.info(f'Training {args.finetuned_model_name} model...')
        if '-cased' in args.pretrained_model_name:
            tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=False)
        else:
            tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=True)

        data_processor = CLaREDataProcessor(args,tokenizer)
        train_samples = data_processor.get_train_sample()

        processed_train_samples  = [data_processor.encoder(item) for item in train_samples]
        print('预处理完毕')
        eval_samples = data_processor.get_dev_sample()
        train_dataset =CLaREDataset(processed_train_samples, data_processor, tokenizer, args, mode='train')
        eval_dataset = CLaREDataset(eval_samples, data_processor, tokenizer, args, mode='eval')
        test_samples = data_processor.get_test_sample()
        test_dataset =CLaREDataset(test_samples, data_processor, tokenizer, args, mode='test')

        model = CLaREModel(model_class, args, data_processor.schema)
        trainer = CLaRETrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            test_dataset = test_dataset,
                            logger=logger)

        global_step, best_step = trainer.train()
        
        
        
    if args.do_predict:
        load_dir = os.path.join(args.output_dir)
        logger.info(f'load tokenizer from {load_dir}')
        if '-cased' in args.pretrained_model_name:
            tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=False)
        else:
            tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=True)

        data_processor = CLaREDataProcessor(args,tokenizer)
        test_samples = data_processor.get_test_sample()
        test_dataset =CLaREDataset(test_samples, data_processor, tokenizer, args, mode='test')
        model = CLaREModel(model_class, args, data_processor.schema)
        trainer = CLaRETrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, test_dataset=test_dataset,logger=logger)
        trainer.load_checkpoint()
        # trainer.predict(mode='test_dev')
        trainer.predict(mode='test')
        


if __name__ == '__main__':
    main()





