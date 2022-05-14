from fastapi import FastAPI ,Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, HTTPException, status
import uvicorn
from webapp.routers import login,predict
import os
os.makedirs("logs",exist_ok=True)


app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"], 
    allow_headers=["*"],
    max_age=2 
    )
app.include_router(login.router)
app.include_router(predict.router)



if __name__ == '__main__':
    uvicorn.run(app)

