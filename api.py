import os
import json
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from model import is_chinese, generate_input
import datetime

import torch

gpu_number = 0
tokenizer = AutoTokenizer.from_pretrained("/data/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/visualglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

app = FastAPI()
@app.post('/')
async def visual_glm(request: Request):
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

    is_zh = is_chinese(input_text)
    input_data = generate_input(input_text, input_image_encoded, history, input_para)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    answer, history = model.chat(tokenizer, input_image, input_text, history, max_length=gen_kwargs['max_length'],
                                               top_p=gen_kwargs['top_p'],
                                               temperature=gen_kwargs['temperature'])
        
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    return response


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"test"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"258258258"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/users/me")
def read_current_user(username: str = Depends(get_current_username)):
    return {"username": username}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
