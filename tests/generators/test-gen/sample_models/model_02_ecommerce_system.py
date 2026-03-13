"""
BUML Model Example 2: E-commerce System
A simple online shopping system with customers, products, and orders
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation,Constraint
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
pending_lit = EnumerationLiteral(name="PENDING")
processing_lit = EnumerationLiteral(name="PROCESSING")
shipped_lit = EnumerationLiteral(name="SHIPPED")
delivered_lit = EnumerationLiteral(name="DELIVERED")
cancelled_lit = EnumerationLiteral(name="CANCELLED")

order_status_enum = Enumeration(
    name="OrderStatus",
    literals={pending_lit, processing_lit, shipped_lit, delivered_lit, cancelled_lit}
)

# =============================================================================
# 2. Define Customer Attributes
# =============================================================================
customer_id_prop = Property(name="customerId", type=StringType, multiplicity=Multiplicity(1, 1))
username_prop = Property(name="username", type=StringType, multiplicity=Multiplicity(1, 1))
password_prop = Property(name="password", type=StringType, multiplicity=Multiplicity(1, 1))
address_prop = Property(name="address", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Product Attributes
# =============================================================================
product_id_prop = Property(name="productId", type=StringType, multiplicity=Multiplicity(1, 1))
product_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
price_prop = Property(name="price", type=FloatType, multiplicity=Multiplicity(1, 1))
stock_prop = Property(name="stock", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Order Attributes
# =============================================================================
order_id_prop = Property(name="orderId", type=StringType, multiplicity=Multiplicity(1, 1))
order_date_prop = Property(name="orderDate", type=StringType, multiplicity=Multiplicity(1, 1))
total_amount_prop = Property(name="totalAmount", type=FloatType, multiplicity=Multiplicity(1, 1))
order_status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Customer Methods
# =============================================================================
login_method = Method(
    name="login",
    parameters=[Parameter(name="inputPassword", type=StringType)],
    code="""
def login(self, inputPassword):
    if self.password == inputPassword:
        print(f"Customer {self.username} logged in successfully")
        return True
    else:
        print("Invalid password")
        return False
"""
)

update_address_method = Method(
    name="updateAddress",
    parameters=[Parameter(name="newAddress", type=StringType)],
    code="""
def updateAddress(self, newAddress):
    old_address = self.address
    self.address = newAddress
    print(f"Address updated from '{old_address}' to '{newAddress}'")
"""
)

get_profile_method = Method(
    name="getProfile",
    parameters=[],
    code="""
def getProfile(self):
    return f"Customer: {self.username} (ID: {self.customerId})\\nAddress: {self.address}"
"""
)

# =============================================================================
# 6. Define Product Methods
# =============================================================================
check_stock_method = Method(
    name="checkStock",
    parameters=[Parameter(name="quantity", type=IntegerType)],
    code="""
def checkStock(self, quantity):
    if self.stock >= quantity:
        print(f"Product '{self.name}' has sufficient stock")
        return True
    else:
        print(f"Insufficient stock for '{self.name}'. Available: {self.stock}")
        return False
"""
)

reduce_stock_method = Method(
    name="reduceStock",
    parameters=[Parameter(name="quantity", type=IntegerType)],
    code="""
def reduceStock(self, quantity):
    if self.stock >= quantity:
        self.stock -= quantity
        print(f"Stock reduced by {quantity}. New stock: {self.stock}")
    else:
        print(f"Cannot reduce stock. Current stock: {self.stock}")
"""
)

get_total_value_method = Method(
    name="getTotalValue",
    parameters=[],
    code="""
def getTotalValue(self):
    total = self.price * self.stock
    print(f"Total inventory value for '{self.name}': ${total:.2f}")
    return total
"""
)

# =============================================================================
# 7. Define Order Methods
# =============================================================================
place_order_method = Method(
    name="placeOrder",
    parameters=[Parameter(name="value", type=StringType)],
    code="""
def placeOrder(self,value:str):
    self.status = value
    print(f"Order {self.orderId} placed on {self.orderDate}")
    print(f"Total amount: ${self.totalAmount:.2f}")
"""
)

update_status_method = Method(
    name="updateStatus",
    parameters=[Parameter(name="newStatus", type=StringType)],
    code="""
def updateStatus(self, newStatus):
    old_status = self.status
    self.status = newStatus
    print(f"Order {self.orderId} status updated: {old_status} -> {newStatus}")
"""
)

calculate_tax_method = Method(
    name="calculateTax",
    parameters=[Parameter(name="taxRate", type=FloatType)],
    code="""
def calculateTax(self, taxRate):
    tax = self.totalAmount * taxRate
    print(f"Tax for order {self.orderId}: ${tax:.2f}")
    return tax
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
customer_class = Class(
    name="Customer",
    attributes={customer_id_prop, username_prop, password_prop, address_prop},
    methods={login_method, update_address_method, get_profile_method}
)

product_class = Class(
    name="Product",
    attributes={product_id_prop, product_name_prop, price_prop, stock_prop},
    methods={check_stock_method, reduce_stock_method, get_total_value_method}
)

order_class = Class(
    name="Order",
    attributes={order_id_prop, order_date_prop, total_amount_prop, order_status_prop},
    methods={place_order_method, update_status_method, calculate_tax_method}
)


constraint_1: Constraint = Constraint(
    name="constraint_1",
    context=order_class,
    expression="context Order::placeOrder(value:string) post placeOrder: self.status = value",
    language="OCL"
)
constraint_2: Constraint = Constraint(
    name="constraint_2",
    context=order_class,
    expression="context Order::updateStatus(value:string) post status: self.status = value",
    language="OCL"
)
# =============================================================================
# 9. Define Associations
# =============================================================================
# Customer --< Order (one customer can have many orders)
customer_end = Property(
    name="customer",
    type=customer_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
customer_orders_end = Property(
    name="orders",
    type=order_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
customer_order_assoc = BinaryAssociation(
    name="Places",
    ends={customer_end, customer_orders_end}
)

# Order >--< Product (many-to-many: orders contain multiple products)
order_end = Property(
    name="orders",
    type=order_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
products_end = Property(
    name="products",
    type=product_class,
    multiplicity=Multiplicity(1, "*"),
    is_navigable=True
)
order_product_assoc = BinaryAssociation(
    name="Contains",
    ends={order_end, products_end}
)
# =============================================================================
# 10. Build the DomainModel
# =============================================================================
ecommerce_model = DomainModel(
    name="EcommerceSystem",
    types={customer_class, product_class, order_class, order_status_enum},
    associations={customer_order_assoc, order_product_assoc},
    constraints={constraint_1,constraint_2}
)

print("✓ E-commerce System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in ecommerce_model.get_classes()]}")
print(f"  Associations: {[a.name for a in ecommerce_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=ecommerce_model, output_dir="output_ecomerce")
python_gen.generate()

from besser.generators.testgen.test_generator import TestGenerator
generator = TestGenerator(model=ecommerce_model, output_dir="output_ecomerce")
generator.generate()