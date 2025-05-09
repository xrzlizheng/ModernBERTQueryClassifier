import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv('../.env')
hf_token = os.getenv('hf_token')
api_key = os.getenv('api_key')

# 加载数据集
dataset = pd.read_excel('./data/dataset.xlsx')

# 显示数据集统计信息
print(f"Total questions: {len(dataset)}")
print(f"Relevant questions: {dataset['is_relevant'].sum()}")
print(f"Non-relevant questions: {len(dataset) - dataset['is_relevant'].sum()}")

# 数据集审查和清理功能可以在这里实现
# 例如：检查重复项、异常值、标签分布等

# 保存审查后的数据集
dataset.to_excel('./data/rev_guardrail_r1.xlsx', index=False)