from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

generator = pipeline("text-generation", model="distilgpt2")

@app.route("/", methods=["GET"])
def home():
    return "WhereismyCAR AI is running."

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    user_input = data.get("text", "")

    prompt = (
        f"你是一个拼车助手。所有人都要在 4:00pm 前抵达 Richmond Nature Park。\n"
        "每个人填写了：姓名、是否有车、出发地（具体地址）、是否能带人（几人）。\n"
        "你需要：将无车的人分配到合理的司机车上，并为司机推荐一个合理的出发时间。\n\n"
        f"成员信息如下：\n{user_input}\n\n"
        "请用以下格式输出：\n"
        "🚗 Lisa：从 XX 出发（建议出发时间：3:00pm）\n→ 搭载：Anna（XX），Jay（XX）\n"
    )

    result = generator(prompt, max_length=350, do_sample=True, temperature=0.7)
    return jsonify({"result": result[0]["generated_text"].replace(prompt, "").strip()})
