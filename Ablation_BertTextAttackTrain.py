#coding=utf-8

#--do_train --bert_model "bert-base-uncased" --model_type "MatchModel" --train_file "data/wikipassageQA/train.tsv" --dev_file "data/wikipassageQA/dev.tsv" --test_file "data/wikipassageQA/test.tsv" --do_lower_case --learning_rate 2e-6 --gpu 0 --epochs 5 --batch_size 2 --accumulate 1 --max_length 100 --shuffle --seed 1024 --save_dir "result/fgm"

#--do_train \
# --bert_model  "bert-base-uncased" \
# --model_type "MatchModel" \
# --train_file "data/wikipassageQA/train.tsv" \
# --dev_file "data/wikipassageQA/dev.tsv" \
# --test_file "data/wikipassageQA/test.tsv" \
# --attacked_file "data/AttackedText/rate1_pos_text.txt"
# --do_lower_case \
# --learning_rate 2e-6 \
# --gpu 0 \
# --epochs 5 \
# --batch_size 2 \
# --accumulate 1 \
# --max_length 100 \
# --shuffle
# --seed 1024 \
# --save_dir "result/test"

from parser1 import args
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os, random
import glob
import torch

import numpy as np
from tqdm import tqdm

# from all_datasets import DataBert
# from AttackDataset import AttackedData

from all_datasets.BertDataset import TrainData

from utils.metrics import mrr, map
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
    # ??????????????????
    train_data = TrainData(data_file=args.train_file,
                          doc_file=doc_file,
                          tokenizer=tokenizer,
                           max_length=args.max_length,
                           attacked_file=args.attacked_file
                          )
    train_dataLoader = DataLoader(dataset=train_data,
                                batch_size=args.batch_size,
                                shuffle= args.shuffle)


    print('train_data:', len(train_data))
    # ????????? optimizer???scheduler
    t_total = len(train_dataLoader) * args.epochs
    num_warmup_steps = int(args.warmup_steps * t_total)
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
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    # apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # ???????????? optimizer???scheduler
    checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
            amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # ????????????
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataLoader))
    logger.debug("  Num Epochs = %d", args.epochs)
    logger.debug("  Set_Batch size = %d", args.batch_size)
    logger.debug("  Real_Batch_size = %d", args.batch_size * args.accumulate)
    logger.debug("  Loss_rate_ = " + str(args.loss_rate))
    logger.debug("  Shuffle = " + str(args.shuffle))
    logger.debug("  warmup_steps = " + str(num_warmup_steps))

    # ???????????????????????????0??????
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1
    logger.debug("  Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []


        step = 0
        attack_batch_count = 0
        for batch, batch_attack in tqdm(train_dataLoader, desc="Iteration", total=len(train_dataLoader)):
            # ??????tensor gpu??????
            model.zero_grad()
            batch = tuple(t.to('cuda') for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask,
                            labels=labels)

            loss_clean = outputs[0]

            batch_attack = tuple(t.to('cuda') for t in batch_attack)

            input_ids2, token_type_ids2, attention_mask2, labels2 = batch_attack


            outputs_attack = model(input_ids=input_ids2.long(),
                                   token_type_ids=token_type_ids2.long(),
                                   attention_mask=attention_mask2,
                                   labels=labels2)


            loss_adv = outputs_attack[0]


            loss = loss_adv + loss_clean

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            #print(loss_clean.item(), loss_adv.item())




            epoch_loss.append(loss.item())


            optimizer.step()
            scheduler.step()

                
            step += 1
            if step % 500 == 0:
              logger.debug("loss:"+str(np.array(epoch_loss).mean()))
              logger.debug('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))

            # ????????????
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
        logger.debug("Use Attack Example Batch:" + str(attack_batch_count))
        # eval dev
        eval_loss, eval_map, eval_mrr = evaluate(model, tokenizer, eval_file=args.dev_file,
                                                checkpoint=epoch,
                                                output_dir=output_dir)
        # eval test
        test_eval_loss, test_eval_map, test_eval_mrr = evaluate(model, tokenizer, eval_file=args.test_file,
                                                                checkpoint=epoch,
                                                                output_dir=output_dir)

        # ???????????? + ????????????
        logger.info('???DEV ???Train Epoch %d: train_loss=%.4f, map=%.4f, mrr=%.4f' % (
        epoch, np.array(epoch_loss).mean(), eval_map, eval_mrr))
        logger.info('???TEST???Train Epoch %d: train_loss=%.4f, map=%.4f, mrr=%.4f' % (
        epoch, np.array(epoch_loss).mean(), test_eval_map, test_eval_mrr))


def evaluate(model, tokenizer, eval_file, checkpoint, output_dir=None):
    eval_data = TrainData(data_file=eval_file,
                         doc_file=doc_file,
                         max_length=args.max_length,
                         tokenizer=tokenizer,
                         attacked_file=None)

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
                            attention_mask=attention_mask,
                            labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    # ????????????
    start = 0

    for key in eval_data.docs_keys:
        end = start + len(eval_data.docs[key])
        maps.append(map(all_labels[start: end], all_logits[start: end]))
        mrrs.append(mrr(all_labels[start: end], all_logits[start: end]))
        start = end

    return np.array(loss).mean(), np.array(maps).mean(), np.array(mrrs).mean()


if __name__ == "__main__":

    # ??????????????????
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = getLogger(__name__, os.path.join(args.save_dir, 'log.txt'))

    if args.do_train:
        # train??? ??????????????????checkpoint????????????
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
        # ??????
        train(model, tokenizer, checkpoint)

    else:
        # eval???????????????
        checkpoint = args.checkpoint
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir,
                                                  do_lower_case=args.do_lower_case,
                                                  )
        model = BertMatchModel.from_pretrained(checkpoint_dir)
        model.to('cuda')
        # ??????
        eval_loss, eval_map, eval_mrr = evaluate(model, tokenizer, eval_file=args.test_file,
                                                checkpoint=checkpoint)
        logger.debug('Evaluate Epoch %d: map=%.4f, mrr=%.4f' % (checkpoint, eval_map, eval_mrr))
