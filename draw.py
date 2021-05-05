import matplotlib.pyplot as plt

#fig = plt.figure(figsize=(2, 100))
# ax = fig.add_subplot(000)
# 没计算的用75标记
plt.axis([1,2.25,75,82])
BERT_base_MAPS=[77.31, 77.05, 76.88, 76.79, 76.71, 75]
Roberta_base_MAPS = [77.57, 78.93, 78.49, 78.49, 79.03, 75]
BERT_large_MAPS=[78.96, 78.93, 78.89, 78.62, 79.01, 78.73]
Roberta_large_MAPS=[79.67, 78.93, 79.55, 80.05, 79.42, 75]
# MRRS=[83.18, 84.28, 83.92, 83.79, 83.38, 83.54]
x=[1, 1.25, 1.5, 1.75, 2, 2.25]

plt.plot(x, BERT_base_MAPS)
plt.plot(x, Roberta_base_MAPS)
plt.plot(x, BERT_large_MAPS)
plt.plot(x, Roberta_large_MAPS)
plt.title("BERT-base", fontsize=20)
plt.xlabel("loss increase ratio:" + chr(949), fontsize=12)
plt.ylabel("MAP & MRR", fontsize=12)
plt.show()
plt.show()
