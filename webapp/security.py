from dotenv import load_dotenv
import os 
from passlib.context import CryptContext
from fastapi import  HTTPException, status
from jose import JWTError,jwt 
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends
import logging
from DeepImageSearch.utils.allutils import SetUpLogging

SetUpLogging("config/logging.yaml").setup_logging()
logger = logging.getLogger("api.applogin")


credentials_user_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
            )

credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
            )



class EnvVariables:
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path)
    SECRET_KEY = os.environ.get('SECRET_KEY')
    ALGORITHM = os.environ.get('ALGORITHM')
    PASSWORD = os.environ.get('PASSWORD')
    USER1 = os.environ.get('USER1')
    USER2 = os.environ.get('USER2')
    oaut2_scheme = OAuth2PasswordBearer(tokenUrl='login')


class secure:

    @staticmethod
    def verify_password(plain_password:str):
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, EnvVariables.PASSWORD)

    @staticmethod
    def create_access_token(username:str,password:str):

        if username not in (EnvVariables.USER1,EnvVariables.USER2):
            logger.critical(f"Anonymous User try to logging username - {username} and password - {password}")
            raise credentials_user_exception
        
        if not secure.verify_password(password):
            logger.critical(f"Anonymous User try to logging {username} and password {password}")
            raise credentials_user_exception
        encoded_to = {"username":username}
        encoded_jwt = jwt.encode(encoded_to,EnvVariables.SECRET_KEY,algorithm=EnvVariables.ALGORITHM)
        logger.warn("successful user logging")
        return encoded_jwt
    
    @staticmethod
    def verify_token(token:str):
        try:
            jwt.decode(token,EnvVariables.SECRET_KEY,algorithms=EnvVariables.ALGORITHM)
        except JWTError:
            raise credentials_exception

    @staticmethod
    def get_current_user(data:str=Depends(EnvVariables.oaut2_scheme)):
        return secure.verify_token(data)
        




