import time as time_module
import logging
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from routers import bill as bill_router
from routers import bill_methods as bill_methods_router
from routers import reservedroom as reservedroom_router
from routers import booking as booking_router
from routers import booking_methods as booking_methods_router
from routers import room as room_router
from routers import person as person_router
from routers import employee as employee_router
from routers import guest as guest_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Class_Diagram API",
    description="Auto-generated REST API with full CRUD operations, relationship management, and advanced features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System", "description": "System health and statistics"},
        {"name": "Bill", "description": "Operations for Bill entities"},
        {"name": "Bill Relationships", "description": "Manage Bill relationships"},
        {"name": "Bill Methods", "description": "Execute Bill methods"},
        {"name": "ReservedRoom", "description": "Operations for ReservedRoom entities"},
        {"name": "ReservedRoom Relationships", "description": "Manage ReservedRoom relationships"},
        {"name": "Booking", "description": "Operations for Booking entities"},
        {"name": "Booking Relationships", "description": "Manage Booking relationships"},
        {"name": "Booking Methods", "description": "Execute Booking methods"},
        {"name": "Room", "description": "Operations for Room entities"},
        {"name": "Room Relationships", "description": "Manage Room relationships"},
        {"name": "Person", "description": "Operations for Person entities"},
        {"name": "Person Relationships", "description": "Manage Person relationships"},
        {"name": "Employee", "description": "Operations for Employee entities"},
        {"name": "Employee Relationships", "description": "Manage Employee relationships"},
        {"name": "Guest", "description": "Operations for Guest entities"},
        {"name": "Guest Relationships", "description": "Manage Guest relationships"},
    ]
)

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################
#
#   Middleware
#
############################################

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time_module.time()
    response = await call_next(request)
    process_time = time_module.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

############################################
#
#   Exception Handlers
#
############################################

# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": str(exc),
            "detail": "Invalid input data provided"
        }
    )


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    """Handle database integrity errors."""
    logger.error(f"Database integrity error: {exc}")

    # Extract more detailed error information
    error_detail = str(exc.orig) if hasattr(exc, 'orig') else str(exc)

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "error": "Conflict",
            "message": "Data conflict occurred",
            "detail": error_detail
        }
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """Handle general SQLAlchemy errors."""
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "Database operation failed",
            "detail": "An internal database error occurred"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format.

    `detail` carries the endpoint's actual message — clients (including the
    generated frontend dialog) read it to show the user what went wrong.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            "message": exc.detail,
            "detail": exc.detail
        }
    )

############################################
#
#   Routers
#
############################################

app.include_router(bill_router.router)
app.include_router(bill_methods_router.router)
app.include_router(reservedroom_router.router)
app.include_router(booking_router.router)
app.include_router(booking_methods_router.router)
app.include_router(room_router.router)
app.include_router(person_router.router)
app.include_router(employee_router.router)
app.include_router(guest_router.router)

############################################
#
#   Global API endpoints
#
############################################

@app.get("/", tags=["System"])
def root():
    """Root endpoint - API information"""
    return {
        "name": "Class_Diagram API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", tags=["System"])
def health_check(database: Session = Depends(get_db)):
    """Health check endpoint for monitoring.

    Actually touches the database — a health check that hardcodes
    "connected" is worse than none.
    """
    from datetime import datetime
    from sqlalchemy import text as _sql_text
    try:
        database.execute(_sql_text("SELECT 1"))
        db_status = "connected"
    except Exception:
        raise HTTPException(status_code=503, detail={
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
        })
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
    }


@app.get("/statistics", tags=["System"])
def get_statistics(database: Session = Depends(get_db)):
    """Get database statistics for all entities"""
    stats = {}
    stats["bill_count"] = database.query(Bill).count()
    stats["reservedroom_count"] = database.query(ReservedRoom).count()
    stats["booking_count"] = database.query(Booking).count()
    stats["room_count"] = database.query(Room).count()
    stats["person_count"] = database.query(Person).count()
    stats["employee_count"] = database.query(Employee).count()
    stats["guest_count"] = database.query(Guest).count()
    stats["total_entities"] = sum(stats.values())
    return stats


############################################
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)