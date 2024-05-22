from datetime import datetime, date
from typing import List, Optional, Union,Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class ReparationCreate(BaseModel):
    date_set: date
    description: str

    use_id: int

            

 

class LifecycleStageCreate(BaseModel):
    start: date
    end: date

    rawmaterials_id: List[int]

            

    productpassports_id: List[int]

            

 

class DesignCreate(LifecycleStageCreate):

    pass
 

class UseCreate(LifecycleStageCreate):

            

    pass
 

class ManufactureCreate(LifecycleStageCreate):

    pass
 

class DistributionCreate(LifecycleStageCreate):

    pass
 

class RawMaterialCreate(BaseModel):
    name: str

    lifecyclestages_id: List[int]

            

 

class ProductPassportCreate(BaseModel):
    product_name: str
    brand: str
    code: str

    lifecyclestages_id: List[int]

            

 

