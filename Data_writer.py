import pandas as pd
import numpy as np

def data_writer(list, fileName):
    data = pd.DataFrame(list)
    # writer = pd.ExcelWriter(fileName)  # 写入Excel文件
    data.to_csv(fileName, sep=',')  # ‘page_1’是写入excel的sheet名
    # writer.save()
    # writer.close()

# list = np.array([[[1, 2], [2, 3]], [[1, 2], [2, 3]]])
# data_writer(list, './ceshi.xlsx', 'test')