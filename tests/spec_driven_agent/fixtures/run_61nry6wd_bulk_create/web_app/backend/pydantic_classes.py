# Excerpt of gpt-5.6-terra-61nry6wd: the model made totalAmountDue, settled
# and issuedDate server-owned by removing them from BillCreate.
from pydantic import BaseModel


class BillCreate(BaseModel):
    billNumber: str
    booking: int  # 1:1 Relationship (mandatory)
