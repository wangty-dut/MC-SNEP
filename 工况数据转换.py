
import pandas as pd

# 读取Excel文件
df = pd.read_excel('汇总.xlsx')

# 定义条件和对应的转换结果
conditions = {7: 4, 5: 4, 4: 2, 6: 2, 3: 3, 1: 2, 2: 2}

# 根据条件进行转换
df['工况变'] = df['工况原'].map(conditions)

# 保存修改后的结果到Excel文件
df.to_excel('汇总.xlsx', index=False)
