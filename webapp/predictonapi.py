from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()


class ClientInput(BaseModel):
    image: bytes
    framwork: str 
    ccbr:bool = False 

class ClientOutput(BaseModel):
    image: bytes
    price: str 
    rattings: str 
    NoOfPurchase: str
    redict_link: str

@router.put("/predict",response_model=ClientOutput)
def predict(file:ClientInput):
    pass
    

