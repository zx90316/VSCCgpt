import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 指定模型名称，可以根据需要更改
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# 加载预训练的句子嵌入模型
print("正在加载嵌入模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("嵌入模型加载完成。")

# 读取手册内容
print("正在读取手册内容...")
with open('manual.txt', 'r', encoding='utf-8') as f:
    manual_text = f.read()
print("手册内容读取完成。")

# 分割手册为章节（以两个换行符为分隔符）
sections = manual_text.strip().split('\n\n')
print(f"共分割出 {len(sections)} 个章节。")

# 定义嵌入函数
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# 生成并保存嵌入
manual_embeddings = []
print("正在生成嵌入向量...")
for idx, section in enumerate(sections):
    embedding = embed_text(section)
    manual_embeddings.append(embedding)
    print(f"已生成第 {idx+1}/{len(sections)} 个章节的嵌入")

# 将嵌入向量转换为 NumPy 数组
manual_embeddings = np.array(manual_embeddings)
print(f"嵌入向量形状：{manual_embeddings.shape}")

# 保存嵌入向量和章节内容
np.save('manual_embeddings.npy', manual_embeddings)
print("嵌入向量已保存为 'manual_embeddings.npy'。")

with open('sections.txt', 'w', encoding='utf-8') as f:
    for section in sections:
        # 去除换行符，确保每个章节占一行
        cleaned_section = section.replace('\n', ' ')
        f.write(cleaned_section + '\n')
print("章节内容已保存为 'sections.txt'。")

print("所有嵌入已生成并保存完成。")
