import os
import json
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from model import process_image
import datetime
import torch

gpu_number = 0
tokenizer = AutoTokenizer.from_pretrained("/data/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/visualglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


app = FastAPI()
@app.post('/')
def visual_glm(request: Request):
    json_post_raw = await request.json()
    print("Start to process request")

    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)
    
    input_text, input_image_encoded, history = request_data['text'], request_data['image'], request_data['history']
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)


    image_path = process_image(input_image_encoded)
    answer, history = model.chat(tokenizer, image_path, input_text, history, max_length=input_para['max_length'],
                                               top_p=input_para['top_p'],
                                               temperature=input_para['temperature'])
        
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
