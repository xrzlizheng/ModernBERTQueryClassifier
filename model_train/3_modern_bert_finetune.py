import os
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfFolder, login
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)


def batch_predict(df, text_column, batch_size=16):
    """
    对 DataFrame 中的文本列进行批量推理
    
    参数:
        df: 包含文本的 pandas DataFrame
        text_column: 包含待分类文本的列名
        batch_size: 一次处理的样本数量
    
    返回:
        追加了预测结果的 DataFrame
    """
    from tqdm.auto import tqdm
    results = []
    texts = df[text_column].tolist()
    total_batches = (len(texts) + batch_size - 1) // batch_size
 
    # 分批处理以避免内存问题
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Processing batches"):

        batch_texts = texts[i:i+batch_size]
        batch_preds = classifier(batch_texts)
        results.extend(batch_preds)
    
    # 创建结果列
    df['predicted_label'] = [pred['label'] for pred in results]
    df['confidence'] = [pred['score'] for pred in results]
    
    return df


load_dotenv('../.env')
hf_token = os.getenv('hf_token')
api_key = os.getenv('api_key')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}


login(token=hf_token, add_to_git_credential=True) 

# 加载数据集
dataset_r1 = pd.read_excel('./data/rev_guardrail_r1.xlsx')
dataset = pd.read_excel('./data/dataset.xlsx') 
print(dataset['is_relevant'].value_counts())
dataset['is_relevant'] = dataset_r1['r1_review_extracted'].to_list()
dataset = dataset.dropna()
print(dataset['is_relevant'].value_counts())

# 准备模型训练
dataset = dataset.rename(columns={'is_relevant':'label'})  # 修正拼写错误: 'lable' -> 'label'
# 在使用分词器之前先加载它
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512  # 设置 model_max_length 为 512

# 将 pandas DataFrame 转换为 Hugging Face Dataset
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
hf_dataset = Dataset.from_pandas(dataset)

# 创建一个类似于 load_dataset 返回的数据集字典
# 分词辅助函数
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")

tokenized_dataset = {}
tokenized_dataset["train"] = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset["test"] = test_dataset.map(tokenize, batched=True, remove_columns=["text"])

print(tokenized_dataset["train"].features.keys())
# 将显示类似这样的内容: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# 加载预训练模型
model_id = "answerdotai/ModernBERT-base"

unique_labels = sorted(train_df['label'].unique().tolist())
num_labels = len(unique_labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(unique_labels):
    label2id[str(label)] = str(i)
    id2label[str(i)] = str(label)
 
# 从 huggingface.co/models 下载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 保存模型
model_path = "./modernbert-llm-router"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# 加载保存的模型进行推理
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 在测试集上进行预测
test_df = batch_predict(test_df, "text")
print(test_df[["text", "label", "predicted_label", "confidence"]].head())

# 计算准确率
accuracy = (test_df["label"] == test_df["predicted_label"]).mean()
print(f"Accuracy: {accuracy:.4f}")