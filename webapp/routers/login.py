from fastapi import APIRouter,Depends
from fastapi.security import  OAuth2PasswordRequestForm
from webapp.security import secure
import logging
from DeepImageSearch.utils.allutils import SetUpLogging

SetUpLogging("config/logging.yaml").setup_logging()
logger = logging.getLogger("api.applogin")

router = APIRouter(
    prefix="/login",
    tags=["LogIn"]
)

@router.post("/")
def login(request:OAuth2PasswordRequestForm=Depends()):
    access_token = secure.create_access_token(request.username,request.password)
    logger.info("Loggin successful")
    return {"access_token": access_token, "token_type": "bearer"}



