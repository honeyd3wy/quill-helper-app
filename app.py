from flask import Flask, request, render_template
from flask_restful import Resource, Api

from pymongo import MongoClient

from torch import tensor
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import os     


class MongoDB:
    def __init__(self):
        self.HOST = 'cluster0.abl5f.mongodb.net'
        self.USER = 'cp1_manager'
        self.PASSWORD = 'pp333'
        self.DATABASE_NAME = 'cp1'
        self.COLLECTION_NAME = 'data_from_app'

    def get_collection(self):
        MONGO_URI = f"mongodb+srv://{self.USER}:{self.PASSWORD}@{self.HOST}/{self.DATABASE_NAME}?retryWrites=true&w=majority"
        # [ssl: certificate_verify_failed] certificate verify failed: certificate has expired (_ssl.c:1131) 오류 아래 명령어로 해결
        # $ pip install certifi --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org
        
        client = MongoClient(MONGO_URI)
        database = client[self.DATABASE_NAME]
        collection = database[self.COLLECTION_NAME]

        return client, database, collection

model = BartForConditionalGeneration.from_pretrained(f'honeyd3wy/kobart-titlenaming-v0.3')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

app = Flask(__name__)
api = Api(app)


# 메인 페이지
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

# about 페이지
@app.route("/about", methods=['GET'])
def about():
    if request.method == 'GET':
        return render_template('about.html')
        

# 결과창
@app.route("/result", methods=['POST'])
def result():
    if request.method == 'POST':
        # input_text 받아오기
        input_text = str(request.form['input_text'])
        version = str(request.form['version'])

        # 버전에 따라 모델, 토크나이저 생성
        # model = BartForConditionalGeneration.from_pretrained(f'honeyd3wy/kobart-titlenaming-v0.3')
        # tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        
        # summarization
        input_ids = tokenizer.encode(input_text)
        input_ids = tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=256, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)

        return render_template("result.html", input=input_text, output=output)

@app.route("/result/completed", methods=['POST'])
def data_add_completed():
    if request.method == 'POST':
        input_text = str(request.form['input'])
        output = str(request.form['output'])
        status = str(request.form['Radiobutton'])

        if status == "1":
            client, database, collection = MongoDB().get_collection()
            record = {'title': output, 'description': input_text}

            collection.insert_one(document=record)
        else:
            pass

        return render_template("completed.html")


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=8000, debug=True)