# Adversial Training for WikiPassageQA

Use Adversial Training for QA retrieval dataset WikiPassageQA

## Running Environments:

+ cuda == 9.2

+ pytorch ==1.5.1 + cu92
+ transformers == 3.3 

## **Running parameters:**

python3 run_FGMSelectedTrain.py --do_train \

--bert_model "bert-base-uncased"   \

--model_type "MatchModel" \

--train_file "data/wikipassageQA/train.tsv"   \

--dev_file "data/wikipassageQA/dev.tsv" \

--test_file "data/wikipassageQA/test.tsv"   \

--attacked_file "attacked_data path"  \

--do_lower_case \

--learning_rate 1.8e-6  \

--gpu 0 \

--epochs 6 \

--batch_size 8\

 --accumulate 1  \

--loss_rate 1.5 \

--shuffle True \

--s1_length 40 \

--s2_length 400  \

--seed 1024 \

--save_dir "result/1.5rate_disorder" \

 --fp16 \

--fptype O2



## File Structure

+ /WordNetAttack  --实现文本攻击所用代码目录
  用词显著性对样本中的词重要性进行排序，利用wordNet和Counter-fitting词向量找出同义词替换
  + dataPreprocess.py  --对原始数据处理成[Q, A , Label]的形式
  + splitDataIntoPieces.py  --将训练数据分为几个部分（为了同时进行攻击并行提高效率）
  + GenerateDataWordsSort.py  --生成每个样本的词重要性顺序
  + GenerateAttackData.py   --仅对样本中最重要的一个词进行替换(最初的测试想法，后来并没用)
  + Word_replace.py  --按照词重要性的顺序，根据一定比例(超参)攻击文本
  + modify_attackdata.py  --对抗样本与原样本的格式略有出入，对其进行调整
  + AggregateAttackData.py  --将各部分攻击样本的数据整合成一个文件
  + testFileNumbers.py  --测试一下生成的样本数量和原样本数量是否相等
  + TestLoss.py  --测试对抗样本与原始样本的Loss差异
+ /all_datasets  --各种模型和训练所需要的dataset格式
  + AttackDataset.py  --文本攻击的数据加载
  + dataset.py  --原始样本的Dataset加载
  + BertDataset.py  --Bert模型原始数据和对抗样本的数据加载
  + RobertaDataset.py --Roberta模型原始数据和对抗样本的数据加载
  + 后面两者合并了前面两者的功能，可以淘汰前两者
+ parser1.py  --参数传输文件
+ run_FGM.py   --FGM对抗训练
+ run_AttackTrain.py  --文本攻击对抗训练
+ run_FGMAttackTrain.py  --FGM+文本攻击对抗训练
+ run_FGMSelectedTrain.py  --不使用所有的对抗样本进行训练，只有对抗样本的损失提升到一定程度时才使用
+ run_baseline_bert.py  --利用新BertDataset新构造的基线训练文件
+ run_BertFGMSelectedTrain.py  --利用新BertDataset新构造的对抗训练文件
+ run_baseline_roberta.py  --利用新RobertaDataset新构造的基线训练文件
+ run_RobertaFGMSelectedTrain.py  --利用新RobertaDataset新构造的对抗训练文件



