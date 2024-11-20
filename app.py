import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
import traceback

app = Flask(__name__)

# 加载嵌入模型
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 加载手册嵌入和章节内容
manual_embeddings = np.load('manual_embeddings.npy')
with open('sections.txt', 'r', encoding='utf-8') as f:
    sections = f.read().splitlines()

# 定义嵌入函数
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# 定义问答函数
def answer_question(question, threshold=0.5):
    question_embedding = embed_text(question).reshape(1, -1)
    similarities = cosine_similarity(question_embedding, manual_embeddings)[0]
    max_similarity = np.max(similarities)

    if max_similarity < threshold:
        return f"抱歉，我無法回答你的問題 {max_similarity*100}"
    else:
        best_match_idx = np.argmax(similarities)
        return sections[best_match_idx]


# 定义路由
@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question']
        answer = answer_question(question)
        return render_template('chat.html', question=question, answer=answer)
    return render_template('chat.html')

if __name__ == '__main__':
    try:
        app.run(debug=False)
    except Exception as e:
        traceback.print_exc()
