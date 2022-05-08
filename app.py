from fastapi import FastAPI ,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from webapp import predictonapi
app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["PUT"], 
    allow_headers=["*"],
    max_age=2 # how mcuh hit api per second
    )
app.include_router(predictonapi.router)
