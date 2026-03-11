from datetime import datetime, date, time

############################################
# Definition of Classes
############################################

class Branch:

    def __init__(self, branchId: str, location: str, branchName: str, maintains: set["Account"] = None):
        self.branchId = branchId
        self.location = location
        self.branchName = branchName
        self.maintains = maintains if maintains is not None else set()
        
    @property
    def branchId(self) -> str:
        return self.__branchId

    @branchId.setter
    def branchId(self, branchId: str):
        self.__branchId = branchId

    @property
    def location(self) -> str:
        return self.__location

    @location.setter
    def location(self, location: str):
        self.__location = location

    @property
    def branchName(self) -> str:
        return self.__branchName

    @branchName.setter
    def branchName(self, branchName: str):
        self.__branchName = branchName

    @property
    def maintains(self):
        return self.__maintains

    @maintains.setter
    def maintains(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Branch__maintains", None)
        self.__maintains = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "branch"):
                    opp_val = getattr(item, "branch", None)
                    
                    if opp_val == self:
                        setattr(item, "branch", None)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "branch"):
                    opp_val = getattr(item, "branch", None)
                    
                    setattr(item, "branch", self)
                    

class Customer:

    def __init__(self, email: str, customerId: str, phoneNumber: str, name: str, owns: set["Account"] = None):
        self.email = email
        self.customerId = customerId
        self.phoneNumber = phoneNumber
        self.name = name
        self.owns = owns if owns is not None else set()
        
    @property
    def phoneNumber(self) -> str:
        return self.__phoneNumber

    @phoneNumber.setter
    def phoneNumber(self, phoneNumber: str):
        self.__phoneNumber = phoneNumber

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def email(self) -> str:
        return self.__email

    @email.setter
    def email(self, email: str):
        self.__email = email

    @property
    def customerId(self) -> str:
        return self.__customerId

    @customerId.setter
    def customerId(self, customerId: str):
        self.__customerId = customerId

    @property
    def owns(self):
        return self.__owns

    @owns.setter
    def owns(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Customer__owns", None)
        self.__owns = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "customer"):
                    opp_val = getattr(item, "customer", None)
                    
                    if opp_val == self:
                        setattr(item, "customer", None)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "customer"):
                    opp_val = getattr(item, "customer", None)
                    
                    setattr(item, "customer", self)
                    

class Account:

    def __init__(self, accountType: str, balance: float, accountNumber: str, branch: "Branch" = None, customer: "Customer" = None, records: set["Transaction"] = None):
        self.accountType = accountType
        self.balance = balance
        self.accountNumber = accountNumber
        self.branch = branch
        self.customer = customer
        self.records = records if records is not None else set()
        
    @property
    def accountType(self) -> str:
        return self.__accountType

    @accountType.setter
    def accountType(self, accountType: str):
        self.__accountType = accountType

    @property
    def balance(self) -> float:
        return self.__balance

    @balance.setter
    def balance(self, balance: float):
        self.__balance = balance

    @property
    def accountNumber(self) -> str:
        return self.__accountNumber

    @accountNumber.setter
    def accountNumber(self, accountNumber: str):
        self.__accountNumber = accountNumber

    @property
    def records(self):
        return self.__records

    @records.setter
    def records(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Account__records", None)
        self.__records = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "account"):
                    opp_val = getattr(item, "account", None)
                    
                    if opp_val == self:
                        setattr(item, "account", None)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "account"):
                    opp_val = getattr(item, "account", None)
                    
                    setattr(item, "account", self)
                    

    @property
    def branch(self):
        return self.__branch

    @branch.setter
    def branch(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Account__branch", None)
        self.__branch = value
        
        # Remove self from old opposite end
        if old_value is not None:
            if hasattr(old_value, "maintains"):
                opp_val = getattr(old_value, "maintains", None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
                
        # Add self to new opposite end
        if value is not None:
            if hasattr(value, "maintains"):
                opp_val = getattr(value, "maintains", None)
                if opp_val is None:
                    setattr(value, "maintains", set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def customer(self):
        return self.__customer

    @customer.setter
    def customer(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Account__customer", None)
        self.__customer = value
        
        # Remove self from old opposite end
        if old_value is not None:
            if hasattr(old_value, "owns"):
                opp_val = getattr(old_value, "owns", None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
                
        # Add self to new opposite end
        if value is not None:
            if hasattr(value, "owns"):
                opp_val = getattr(value, "owns", None)
                if opp_val is None:
                    setattr(value, "owns", set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def deposit(self, amount: float):
        # TODO: Implement deposit method
        pass

    def withdraw(self, amount: float):
        # TODO: Implement withdraw method
        pass

class Transaction:

    def __init__(self, transactionId: str, transactionType: str, amount: float, transactionDate: date, account: "Account" = None):
        self.transactionId = transactionId
        self.transactionType = transactionType
        self.amount = amount
        self.transactionDate = transactionDate
        self.account = account
        
    @property
    def transactionId(self) -> str:
        return self.__transactionId

    @transactionId.setter
    def transactionId(self, transactionId: str):
        self.__transactionId = transactionId

    @property
    def transactionType(self) -> str:
        return self.__transactionType

    @transactionType.setter
    def transactionType(self, transactionType: str):
        self.__transactionType = transactionType

    @property
    def transactionDate(self) -> date:
        return self.__transactionDate

    @transactionDate.setter
    def transactionDate(self, transactionDate: date):
        self.__transactionDate = transactionDate

    @property
    def amount(self) -> float:
        return self.__amount

    @amount.setter
    def amount(self, amount: float):
        self.__amount = amount

    @property
    def account(self):
        return self.__account

    @account.setter
    def account(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Transaction__account", None)
        self.__account = value
        
        # Remove self from old opposite end
        if old_value is not None:
            if hasattr(old_value, "records"):
                opp_val = getattr(old_value, "records", None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
                
        # Add self to new opposite end
        if value is not None:
            if hasattr(value, "records"):
                opp_val = getattr(value, "records", None)
                if opp_val is None:
                    setattr(value, "records", set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)
