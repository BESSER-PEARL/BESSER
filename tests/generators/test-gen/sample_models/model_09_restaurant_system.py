"""
BUML Model Example 9: Restaurant Management System
A simple restaurant system with menu items, tables, and orders
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
appetizer_lit = EnumerationLiteral(name="APPETIZER")
main_course_lit = EnumerationLiteral(name="MAIN_COURSE")
dessert_lit = EnumerationLiteral(name="DESSERT")
beverage_lit = EnumerationLiteral(name="BEVERAGE")

menu_category_enum = Enumeration(
    name="MenuCategory",
    literals={appetizer_lit, main_course_lit, dessert_lit, beverage_lit}
)

# =============================================================================
# 2. Define MenuItem Attributes
# =============================================================================
item_id_prop = Property(name="itemId", type=StringType, multiplicity=Multiplicity(1, 1))
item_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
item_price_prop = Property(name="price", type=FloatType, multiplicity=Multiplicity(1, 1))
item_available_prop = Property(name="available", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Table Attributes
# =============================================================================
table_number_prop = Property(name="tableNumber", type=StringType, multiplicity=Multiplicity(1, 1))
capacity_prop = Property(name="capacity", type=IntegerType, multiplicity=Multiplicity(1, 1))
occupied_prop = Property(name="occupied", type=BooleanType, multiplicity=Multiplicity(1, 1))
location_prop = Property(name="location", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define DineOrder Attributes
# =============================================================================
dine_order_id_prop = Property(name="orderId", type=StringType, multiplicity=Multiplicity(1, 1))
order_time_prop = Property(name="orderTime", type=StringType, multiplicity=Multiplicity(1, 1))
order_total_amount_prop = Property(name="totalAmount", type=FloatType, multiplicity=Multiplicity(1, 1))
dine_order_status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define MenuItem Methods
# =============================================================================
add_to_menu_method = Method(
    name="addToMenu",
    parameters=[],
    code="""
def addToMenu(self):
    self.available = True
    print(f"Menu item added: {self.name}")
    print(f"ID: {self.itemId}, Price: ${self.price:.2f}")
"""
)

update_price_method = Method(
    name="updatePrice",
    parameters=[Parameter(name="newPrice", type=FloatType)],
    code="""
def updatePrice(self, newPrice):
    old_price = self.price
    self.price = newPrice
    print(f"Price updated for {self.name}: ${old_price:.2f} -> ${newPrice:.2f}")
"""
)

set_item_availability_method = Method(
    name="setAvailability",
    parameters=[Parameter(name="status", type=BooleanType)],
    code="""
def setAvailability(self, status):
    self.available = status
    status_text = "available" if status else "sold out"
    print(f"{self.name} is now {status_text}")
"""
)

# =============================================================================
# 6. Define Table Methods
# =============================================================================
reserve_table_method = Method(
    name="reserveTable",
    parameters=[],
    code="""
def reserveTable(self):
    if not self.occupied:
        self.occupied = True
        print(f"Table {self.tableNumber} reserved")
        print(f"Capacity: {self.capacity}, Location: {self.location}")
    else:
        print(f"Table {self.tableNumber} is already occupied")
"""
)

release_table_method = Method(
    name="releaseTable",
    parameters=[],
    code="""
def releaseTable(self):
    self.occupied = False
    print(f"Table {self.tableNumber} is now available")
"""
)

can_accommodate_method = Method(
    name="canAccommodate",
    parameters=[Parameter(name="partySize", type=IntegerType)],
    code="""
def canAccommodate(self, partySize):
    if partySize <= self.capacity and not self.occupied:
        print(f"Table {self.tableNumber} can accommodate {partySize} guests")
        return True
    else:
        print(f"Table {self.tableNumber} cannot accommodate the party")
        return False
"""
)

# =============================================================================
# 7. Define DineOrder Methods
# =============================================================================
place_dine_order_method = Method(
    name="placeOrder",
    parameters=[],
    code="""
def placeOrder(self):
    self.status = "Pending"
    print(f"Order {self.orderId} placed at {self.orderTime}")
    print(f"Total amount: ${self.totalAmount:.2f}")
"""
)

update_dine_order_status_method = Method(
    name="updateStatus",
    parameters=[Parameter(name="newStatus", type=StringType)],
    code="""
def updateStatus(self, newStatus):
    old_status = self.status
    self.status = newStatus
    print(f"Order {self.orderId} status: {old_status} -> {newStatus}")
"""
)

calculate_tip_method = Method(
    name="calculateTip",
    parameters=[Parameter(name="tipPercentage", type=FloatType)],
    code="""
def calculateTip(self, tipPercentage):
    tip = self.totalAmount * tipPercentage
    print(f"Suggested tip ({tipPercentage*100:.0f}%): ${tip:.2f}")
    return tip
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
menu_item_class = Class(
    name="MenuItem",
    attributes={item_id_prop, item_name_prop, item_price_prop, item_available_prop},
    methods={add_to_menu_method, update_price_method, set_item_availability_method}
)

table_class = Class(
    name="Table",
    attributes={table_number_prop, capacity_prop, occupied_prop, location_prop},
    methods={reserve_table_method, release_table_method, can_accommodate_method}
)

dine_order_class = Class(
    name="DineOrder",
    attributes={dine_order_id_prop, order_time_prop, order_total_amount_prop, dine_order_status_prop},
    methods={place_dine_order_method, update_dine_order_status_method, calculate_tip_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# Table --< DineOrder (one table can have many orders)
table_end = Property(
    name="table",
    type=table_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
table_orders_end = Property(
    name="orders",
    type=dine_order_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
table_order_assoc = BinaryAssociation(
    name="HasOrder",
    ends={table_end, table_orders_end}
)

# DineOrder >--< MenuItem (many-to-many: orders include multiple items)
dine_order_end = Property(
    name="orders",
    type=dine_order_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
menu_items_end = Property(
    name="items",
    type=menu_item_class,
    multiplicity=Multiplicity(1, "*"),
    is_navigable=True
)
order_menuitem_assoc = BinaryAssociation(
    name="Includes",
    ends={dine_order_end, menu_items_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
restaurant_model = DomainModel(
    name="RestaurantManagementSystem",
    types={menu_item_class, table_class, dine_order_class, menu_category_enum},
    associations={table_order_assoc, order_menuitem_assoc}
)

print("✓ Restaurant Management System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in restaurant_model.get_classes()]}")
print(f"  Associations: {[a.name for a in restaurant_model.associations]}")

from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=restaurant_model, output_dir="output_restaurant")
python_gen.generate()