"""
BUML Model Example 6: Hotel Reservation System
A simple hotel system with guests, rooms, and reservations
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
single_lit = EnumerationLiteral(name="SINGLE")
double_lit = EnumerationLiteral(name="DOUBLE")
suite_lit = EnumerationLiteral(name="SUITE")
deluxe_lit = EnumerationLiteral(name="DELUXE")

room_type_enum = Enumeration(
    name="RoomType",
    literals={single_lit, double_lit, suite_lit, deluxe_lit}
)

# =============================================================================
# 2. Define Guest Attributes
# =============================================================================
guest_id_prop = Property(name="guestId", type=StringType, multiplicity=Multiplicity(1, 1))
guest_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
email_prop = Property(name="email", type=StringType, multiplicity=Multiplicity(1, 1))
loyalty_points_prop = Property(name="loyaltyPoints", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Room Attributes
# =============================================================================
room_number_prop = Property(name="roomNumber", type=StringType, multiplicity=Multiplicity(1, 1))
room_type_prop = Property(name="roomType", type=StringType, multiplicity=Multiplicity(1, 1))
price_per_night_prop = Property(name="pricePerNight", type=FloatType, multiplicity=Multiplicity(1, 1))
room_available_prop = Property(name="available", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Reservation Attributes
# =============================================================================
reservation_id_prop = Property(name="reservationId", type=StringType, multiplicity=Multiplicity(1, 1))
check_in_date_prop = Property(name="checkInDate", type=StringType, multiplicity=Multiplicity(1, 1))
check_out_date_prop = Property(name="checkOutDate", type=StringType, multiplicity=Multiplicity(1, 1))
total_cost_prop = Property(name="totalCost", type=FloatType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Guest Methods
# =============================================================================
check_in_method = Method(
    name="checkIn",
    parameters=[],
    code="""
def checkIn(self):
    print(f"Guest {self.name} checked in")
    print(f"ID: {self.guestId}, Email: {self.email}")
"""
)

add_loyalty_points_method = Method(
    name="addLoyaltyPoints",
    parameters=[Parameter(name="points", type=IntegerType)],
    code="""
def addLoyaltyPoints(self, points):
    self.loyaltyPoints += points
    print(f"{points} loyalty points added to {self.name}")
    print(f"Total loyalty points: {self.loyaltyPoints}")
"""
)

get_guest_info_method = Method(
    name="getGuestInfo",
    parameters=[],
    code="""
def getGuestInfo(self):
    return f"Guest: {self.name} (ID: {self.guestId})\\nLoyalty Points: {self.loyaltyPoints}"
"""
)

# =============================================================================
# 6. Define Room Methods
# =============================================================================
set_available_method = Method(
    name="setAvailable",
    parameters=[Parameter(name="status", type=BooleanType)],
    code="""
def setAvailable(self, status):
    self.available = status
    status_text = "available" if status else "occupied"
    print(f"Room {self.roomNumber} is now {status_text}")
"""
)

calculate_cost_method = Method(
    name="calculateCost",
    parameters=[Parameter(name="nights", type=IntegerType)],
    code="""
def calculateCost(self, nights):
    total = self.pricePerNight * nights
    print(f"Cost for {nights} night(s) in room {self.roomNumber}: ${total:.2f}")
    return total
"""
)

get_room_details_method = Method(
    name="getRoomDetails",
    parameters=[],
    code="""
def getRoomDetails(self):
    status = "Available" if self.available else "Occupied"
    return f"Room {self.roomNumber} - {self.roomType} (${self.pricePerNight}/night) - {status}"
"""
)

# =============================================================================
# 7. Define Reservation Methods
# =============================================================================
create_reservation_method = Method(
    name="createReservation",
    parameters=[],
    code="""
def createReservation(self):
    print(f"Reservation {self.reservationId} created")
    print(f"Check-in: {self.checkInDate}, Check-out: {self.checkOutDate}")
    print(f"Total cost: ${self.totalCost:.2f}")
"""
)

cancel_reservation_method = Method(
    name="cancelReservation",
    parameters=[],
    code="""
def cancelReservation(self):
    print(f"Reservation {self.reservationId} has been cancelled")
    print(f"Refund amount: ${self.totalCost:.2f}")
"""
)

calculate_nights_method = Method(
    name="calculateNights",
    parameters=[],
    code="""
def calculateNights(self):
    # Simplified calculation - in real system would parse dates
    nights = 3  # placeholder
    print(f"Number of nights: {nights}")
    return nights
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
guest_class = Class(
    name="Guest",
    attributes={guest_id_prop, guest_name_prop, email_prop, loyalty_points_prop},
    methods={check_in_method, add_loyalty_points_method, get_guest_info_method}
)

room_class = Class(
    name="Room",
    attributes={room_number_prop, room_type_prop, price_per_night_prop, room_available_prop},
    methods={set_available_method, calculate_cost_method, get_room_details_method}
)

reservation_class = Class(
    name="Reservation",
    attributes={reservation_id_prop, check_in_date_prop, check_out_date_prop, total_cost_prop},
    methods={create_reservation_method, cancel_reservation_method, calculate_nights_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# Guest --< Reservation (one guest can have many reservations)
guest_end = Property(
    name="guest",
    type=guest_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
guest_reservations_end = Property(
    name="reservations",
    type=reservation_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
guest_reservation_assoc = BinaryAssociation(
    name="Makes",
    ends={guest_end, guest_reservations_end}
)

# Room --< Reservation (one room can have many reservations over time)
room_end = Property(
    name="room",
    type=room_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
room_reservations_end = Property(
    name="bookings",
    type=reservation_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
room_reservation_assoc = BinaryAssociation(
    name="IsReservedBy",
    ends={room_end, room_reservations_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
hotel_model = DomainModel(
    name="HotelReservationSystem",
    types={guest_class, room_class, reservation_class, room_type_enum},
    associations={guest_reservation_assoc, room_reservation_assoc}
)

print("✓ Hotel Reservation System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in hotel_model.get_classes()]}")
print(f"  Associations: {[a.name for a in hotel_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=hotel_model, output_dir="output_hotel")
python_gen.generate()