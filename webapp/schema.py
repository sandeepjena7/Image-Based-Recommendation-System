from pydantic import BaseModel
from typing import Optional,List,Dict

class User(BaseModel):
    image:bytes
    Framework:Optional[str] = "Tensorflow"
    Technique: Optional[str] = 'CCBR'



class  Output(BaseModel):
    pass