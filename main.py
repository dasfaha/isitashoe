from typing import Optional
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastai.vision import *

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name='static')

templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/question2/", response_class=HTMLResponse)
async def read_item(request: Request ):
    return templates.TemplateResponse("question2.html", {'request': request})

@app.get("/camera/", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse("camera.html", {'request': request})

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with  open("this.webp", 'wb') as f:
        f.write(file.file.read())
    learn = load_learner(path= 'model/', file='shoeTrainer.pkl')
    img = open_image(file.file)
    pred_class,pred_idx,outputs = learn.predict(img)
    print(pred_class.obj)
    return {"classification": pred_class.obj}
