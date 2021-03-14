#coding=utf-8
from parser1 import args
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
import os, random
import glob
import torch

import numpy as np
from tqdm import tqdm, trange

from dataset import DataBert
from AttackDataset import AttackedData

from utils.metrics import mrr, map, ndcg
from utils.logger import getLogger

from utils.FGM import FGM


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

if args.seed > -1:
  seed_torch(args.seed)


model_name = 'model.' + args.model_type
BertMatchModel = __import__(model_name, globals(), locals(), [args.model_type]).BertMatchModel

loss_rate = args.loss_rate
logger = None
doc_file='data/wikipassageQA/document_passages.json'


def train(model, tokenizer, checkpoint):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None
    # 训练数据处理
    train_data = DataBert(data_file=args.train_file,
                          doc_file=doc_file,
                          s1_length=args.s1_length,
                          s2_length=args.s2_length,
                          max_length=args.max_length,
                          tokenizer=tokenizer
                          )
    train_dataLoader = DataLoader(dataset=train_data,
                                batch_size=args.batch_size,
                                shuffle= args.shuffle)

    attacked_data = AttackedData(attacked_file=args.attacked_file)  #攻击样本

    attack_dataloader = DataLoader(dataset=attacked_data,
                                   batch_size=args.batch_size,
                                   shuffle=args.shuffle)

    print('train_data:', len(train_data))
    print('attack_data:', len(attacked_data))
    # 初始化 optimizer，scheduler
    t_total = len(train_dataLoader) * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # 读取断点 optimizer、scheduler
    checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
            amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataLoader))
    logger.debug("  Num Epochs = %d", args.epochs)
    logger.debug("  Set_Batch size = %d", args.batch_size)
    logger.debug("  Real_Batch_size = %d", args.batch_size * args.accumulate)
    logger.debug("  Loss_rate_ = " + str(args.loss_rate))
    logger.debug("  Shuffle = " + str(args.shuffle))

    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1
    logger.debug("  Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []


        fgm = FGM(model)
        step = 0
        attack_batch_count = 0
        for batch, batch_attack in tqdm(zip(train_dataLoader, attack_dataloader), desc="Iteration", total=len(train_dataLoader)):
            # 设置tensor gpu运行
            model.zero_grad()
            batch = tuple(t.to('cuda') for t in batch[:4])
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            loss_clean = outputs[0]

            if args.fp16:
                with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss_clean.backward(retain_graph=True) #计算出梯度

            batch_attack = tuple(t.to('cuda') for t in batch_attack)

            input_ids2, token_type_ids2, attention_mask2, labels2 = batch_attack

            fgm.attack()  # 根据梯度进行扰动


            outputs_attack = model(input_ids=input_ids2.long(),
                                   token_type_ids=token_type_ids2.long(),
                                   attention_mask=attention_mask2,
                                   labels=labels2)


            loss_adv = outputs_attack[0]

            if loss_adv > loss_clean * 2:   #如果大于2倍的loss，那就使用FGM+文本攻击
                model.zero_grad()  #先消除梯度，再一起反向传播
                loss = loss_clean + loss_adv  #损失为原始样本和对抗样本之和
                attack_batch_count += 1

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            else:    #如果没有大于2倍loss，就用FGM攻击
                outputs_attack = model(input_ids=input_ids.long(),
                                   token_type_ids=token_type_ids.long(),
                                   labels=labels)

                loss_adv = outputs_attack[0]    #用FGM直接把梯度堆叠上原来的梯度
                loss = loss_adv + loss_clean

                if args.fp16:
                    with amp.scale_loss(loss_adv, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_adv.backward()


            #print(loss_clean.item(), loss_adv.item())




            epoch_loss.append(loss.item())

            fgm.restore()

            optimizer.step()
            scheduler.step()

                
            step += 1

            # 保存模型
        output_dir = args.save_dir + "/checkpoint-" + str(epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.debug("Saving model checkpoint to %s", output_dir)
        if args.fp16:
            torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.debug("Saving optimizer and scheduler states to %s", output_dir)
        logger.debug("Use Attack Example Batch:", str(attack_batch_count))
        # eval dev
        eval_loss, eval_map, eval_mrr = evaluate(model, tokenizer, eval_file=args.dev_file,
                                                checkpoint=epoch,
                                                output_dir=output_dir)
        # eval test
        test_eval_loss, test_eval_map, test_eval_mrr = evaluate(model, tokenizer, eval_file=args.test_file,
                                                                checkpoint=epoch,
                                                                output_dir=output_dir)

        # 输出日志 + 保存日志
        logger.info('【DEV 】Train Epoch %d: train_loss=%.4f, map=%.4f, mrr=%.4f' % (
        epoch, np.array(epoch_loss).mean(), eval_map, eval_mrr))
        logger.info('【TEST】Train Epoch %d: train_loss=%.4f, map=%.4f, mrr=%.4f' % (
        epoch, np.array(epoch_loss).mean(), test_eval_map, test_eval_mrr))


def evaluate(model, tokenizer, eval_file, checkpoint, output_dir=None):
    eval_data = DataBert(data_file=eval_file,
                         doc_file=doc_file,
                         s1_length=args.s1_length,
                         s2_length=args.s2_length,
                         max_length=args.max_length,
                         tokenizer=tokenizer)

    eval_dataLoader = DataLoader(dataset=eval_data,
                                batch_size=args.batch_size,
                                shuffle=False)


    logger.debug("***** Running evaluation {} *****".format(checkpoint))
    logger.debug("  Num examples = %d", len(eval_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    loss = []

    mrrs = []
    maps = []

    all_labels = None
    all_logits = None
    model.eval()

    for batch in tqdm(eval_dataLoader, desc="Evaluating"):
        batch = tuple(t.to('cuda') for t in batch[:4])
        input_ids, token_type_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    # 评价指标
    start = 0

    for key in eval_data.docs_keys:
        end = start + len(eval_data.docs[key])
        maps.append(map(all_labels[start: end], all_logits[start: end]))
        mrrs.append(mrr(all_labels[start: end], all_logits[start: end]))
        start = end

    return np.array(loss).mean(), np.array(maps).mean(), np.array(mrrs).mean()


if __name__ == "__main__":

    # 创建存储目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = getLogger(__name__, os.path.join(args.save_dir, 'log.txt'))

    if args.do_train:
        # train： 接着未训练完checkpoint继续训练
        checkpoint = -1
        for checkpoint_dir_name in glob.glob(args.save_dir + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
            except Exception as e:
                pass
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
        if checkpoint > -1:
            logger.debug("Load Model from {}".format(checkpoint_dir))
        tokenizer = BertTokenizer.from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir,
                                                  do_lower_case=args.do_lower_case
                                                  )
        model = BertMatchModel.from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir)
        model.to('cuda')
        # 训练
        train(model, tokenizer, checkpoint)

    else:
        # eval：指定模型
        checkpoint = args.checkpoint
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir,
                                                  do_lower_case=args.do_lower_case,
                                                  )
        model = BertMatchModel.from_pretrained(checkpoint_dir)
        model.to('cuda')
        # 评估
        eval_loss, eval_map, eval_mrr = evaluate(model, tokenizer, eval_file=args.test_file,
                                                checkpoint=checkpoint)
        logger.debug('Evaluate Epoch %d: map=%.4f, mrr=%.4f' % (checkpoint, eval_map, eval_mrr))
