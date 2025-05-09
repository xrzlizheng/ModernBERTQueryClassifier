import pandas as pd
from tqdm import tqdm
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv('../.env')
hf_token = os.getenv('hf_token')
api_key = os.getenv('api_key')

# 生成基础案例
prompt = """# 物流支持路由器 - 训练数据生成

## 任务
为Spellbots Logistics生成多样化的客户问题，每个问题需被分类为相关（关于我们的物流服务）或不相关（关于其他主题）。

## 输出格式
以如下JSON格式生成数据：
```json
{
  "questions": [
    {
      "text": "When will my package from Spellbots Logistics arrive?",
      "is_relevant": true
    },
    {
      "text": "What's the best recipe for chocolate chip cookies?",
      "is_relevant": false
    }
  ]
}
```

## 指南

### 相关问题（is_relevant = true）
生成关于以下内容的问题：
- 包裹追踪与送达状态
- 配送问题（延误、丢失包裹、损坏物品）
- 运费选项与价格
- 国际运输与海关
- 退货与理赔
- 商业运输解决方案

示例：
- "如何追踪我的Spellbots Logistics包裹？"
- "我的包裹在运输过程中损坏了，如何申请理赔？"
- "你们为国际包裹提供哪些运输选项？"

### 不相关问题（is_relevant = false）
生成明显与Spellbots Logistics服务无关的问题：
- 关于无关主题的常规问题（美食、科技、娱乐等）
- 关于其他快递公司的问题
- 关于物流但与运输/配送服务无关的问题

包含一些“难度较高”的不相关示例，这些问题提及运输或配送但与Spellbots Logistics服务无关。

示例：
- "如何重置我的邮箱密码？"
- "你能推荐一家好的意大利餐厅吗？"
- "如何追踪我的UPS包裹？"（不同公司）
- "你们出售包装材料吗？"（产品问题，不是服务）

### 多样性要求
- 问题长度多样（短、中、长）
- 包含不同客户情绪（中性、紧急、沮丧、困惑）
- 混合不同类型的问题（操作类、状态查询、政策问题）
- 使用真实客户可能使用的语言
- 避免重复或非常相似的问题

## 生成说明
1. 每批生成50个问题（25个相关，25个不相关）
2. 对于相关问题：
   - 仅约30%的问题明确提及“Spellbots Logistics”
   - 其余70%不指明公司名（如“我的包裹在哪里？”而不是“我的Spellbots Logistics包裹在哪里？”）
   - 两种类型均视为相关
3. 对于不相关问题：
   - 明确提及其他快递公司的问题（如“我的UPS包裹何时到？”）应标记为不相关
4. 创建真实客户可能会问的问题

最终数据集应至少包含500个问题，相关与不相关示例均衡。# 物流支持路由器 - 训练数据生成

## 任务
为Spellbots Logistics生成多样化的客户问题，每个问题需被分类为相关（关于我们的物流服务）或不相关（关于其他主题）。

## 输出格式
以如下JSON格式生成数据：
```json
{
  "questions": [
    {
      "text": "When will my package from Spellbots Logistics arrive?",
      "is_relevant": true
    },
    {
      "text": "What's the best recipe for chocolate chip cookies?",
      "is_relevant": false
    }
  ]
}
```

## 指南

### 相关问题（is_relevant = true）
生成关于以下内容的问题：
- 包裹追踪与送达状态
- 配送问题（延误、丢失包裹、损坏物品）
- 运费选项与价格
- 国际运输与海关
- 退货与理赔
- 商业运输解决方案
- 包含部分常见的越狱大模型提问示例

示例：
- "如何追踪我的Spellbots Logistics包裹？"
- "我的包裹在运输过程中损坏了，如何申请理赔？"
- "你们为国际包裹提供哪些运输选项？"

### 不相关问题（is_relevant = false）
生成明显与Spellbots Logistics服务无关的问题：
- 关于无关主题的常规问题（美食、科技、娱乐等）
- 关于其他快递公司的问题
- 关于物流但与运输/配送服务无关的问题

包含一些“难度较高”的不相关示例，这些问题提及运输或配送但与Spellbots Logistics服务无关。

示例：
- "如何重置我的邮箱密码？"
- "你能推荐一家好的意大利餐厅吗？"
- "如何追踪我的UPS包裹？"（不同公司）
- "你们出售包装材料吗？"（产品问题，不是服务）

### 多样性要求
- 问题长度多样（短、中、长）
- 包含不同客户情绪（中性、紧急、沮丧、困惑）
- 混合不同类型的问题（操作类、状态查询、政策问题）
- 使用真实客户可能使用的语言
- 避免重复或非常相似的问题

## 生成说明
1. 每批生成50个问题（25个相关，25个不相关）
2. 对于相关问题：
   - 仅约30%的问题明确提及“Spellbots Logistics”
   - 其余70%不指明公司名（如“我的包裹在哪里？”而不是“我的Spellbots Logistics包裹在哪里？”）
   - 两种类型均视为相关
3. 对于不相关问题：
   - 明确提及其他快递公司的问题（如“我的UPS包裹何时到？”）应标记为不相关
4. 创建真实客户可能会问的问题
5. 除了json响应外不返回任何内容

最终数据集应至少包含50个问题，相关与不相关示例均衡。"""

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/fireworks/models/deepseek-v3",
  "max_tokens": 16384,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.9,
  "messages": [
    {
      "role": "user",
      "content": prompt
    }
  ]
}

# 调用API并获取响应的函数
def get_response(payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# 批量生成问题
all_questions = []

# 你可以取消注释并运行此代码以生成数据集
'''
for i in tqdm(range(20)):
    response = get_response(payload)
    try:
        content = response['choices'][0]['message']['content']
        # 提取JSON部分
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("````)[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        
        # 解析JSON
        data = json.loads(json_str)
        all_questions.extend(data['questions'])
        
        # 每批保存当前状态
        df = pd.DataFrame(all_questions)
        df.to_excel('./data/questions_sample_2k.xlsx', index=False)
        
    except Exception as e:
        print(f"处理第{i}批次时出错: {e}")
        print("响应:", response)
'''

# 加载已生成的问题
df = pd.read_excel('./data/questions_sample_2k.xlsx')

# 展示部分统计信息
print(f"问题总数: {len(df)}")
print(f"相关问题数: {df['is_relevant'].sum()}")
print(f"不相关问题数: {len(df) - df['is_relevant'].sum()}")

# 保存为最终数据集文件
df.to_excel('./data/dataset.xlsx', index=False)