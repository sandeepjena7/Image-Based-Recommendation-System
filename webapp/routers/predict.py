from fastapi import APIRouter,Depends, HTTPException, status
from webapp.security import secure
from webapp import schema
from DeepImageSearch.utils.allutils import util
from typing import List,Dict
from run import RUN
from DeepImageSearch.utils.allutils import SetUpLogging
import logging

SetUpLogging("config/logging.yaml").setup_logging()
logger = logging.getLogger("api.apppredict")
obj1 = RUN('config/config.yaml','params.yaml')

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    dependencies=[Depends(secure.get_current_user)]
                 )

@router.post("/")
def predict(user:schema.User):
    try:
        util.decode_base64(user.image,"uploads/api.png")
        output = obj1.search(Framework=user.Framework,Technique=user.Technique,image="uploads/api.png",usedcase='api')
        logger.info("Sucessful send prediction base64 images")
        return output
    except Exception as e:
        logger.error(e)
        logger.debug(e)
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY
                            ,detail='some error at RUN')
