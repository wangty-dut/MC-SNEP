import matplotlib.pyplot as plt
import pandas as pd

train_loss = pd.read_excel("./loss保存/condition3/trloss.xlsx").values
val_loss = pd.read_excel("./loss保存/condition3/valloss.xlsx").values

train_loss = train_loss[:, 1]
val_loss = val_loss[:, 1]

plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend(loc='best', fontsize=15)
plt.show()