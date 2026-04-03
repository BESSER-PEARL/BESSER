import uvicorn
import os, json
import time as time_module
import logging
from fastapi import Depends, FastAPI, HTTPException, Request, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic_classes import *
from sql_alchemy import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################
#
#   Initialize the database
#
############################################

def init_db():
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/BlogPlatform.db")
    # Ensure local SQLite directory exists (safe no-op for other DBs)
    os.makedirs("data", exist_ok=True)
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal

app = FastAPI(
    title="BlogPlatform API",
    description="Auto-generated REST API with full CRUD operations, relationship management, and advanced features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System", "description": "System health and statistics"},
        {"name": "Comment", "description": "Operations for Comment entities"},
        {"name": "Comment Relationships", "description": "Manage Comment relationships"},
        {"name": "Post", "description": "Operations for Post entities"},
        {"name": "Post Relationships", "description": "Manage Post relationships"},
        {"name": "User", "description": "Operations for User entities"},
        {"name": "User Relationships", "description": "Manage User relationships"},
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
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            "message": exc.detail,
            "detail": f"HTTP {exc.status_code} error occurred"
        }
    )

# Initialize database session
SessionLocal = init_db()
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        logger.error("Database session rollback due to exception")
        raise
    finally:
        db.close()

############################################
#
#   Global API endpoints
#
############################################

@app.get("/", tags=["System"])
def root():
    """Root endpoint - API information"""
    return {
        "name": "BlogPlatform API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for monitoring"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }


@app.get("/statistics", tags=["System"])
def get_statistics(database: Session = Depends(get_db)):
    """Get database statistics for all entities"""
    stats = {}
    stats["comment_count"] = database.query(Comment).count()
    stats["post_count"] = database.query(Post).count()
    stats["user_count"] = database.query(User).count()
    stats["total_entities"] = sum(stats.values())
    return stats


############################################
#
#   BESSER Action Language standard lib
#
############################################


async def BAL_size(sequence:list) -> int:
    return len(sequence)

async def BAL_is_empty(sequence:list) -> bool:
    return len(sequence) == 0

async def BAL_add(sequence:list, elem) -> None:
    sequence.append(elem)

async def BAL_remove(sequence:list, elem) -> None:
    sequence.remove(elem)

async def BAL_contains(sequence:list, elem) -> bool:
    return elem in sequence

async def BAL_filter(sequence:list, predicate) -> list:
    return [elem for elem in sequence if predicate(elem)]

async def BAL_forall(sequence:list, predicate) -> bool:
    for elem in sequence:
        if not predicate(elem):
            return False
    return True

async def BAL_exists(sequence:list, predicate) -> bool:
    for elem in sequence:
        if predicate(elem):
            return True
    return False

async def BAL_one(sequence:list, predicate) -> bool:
    found = False
    for elem in sequence:
        if predicate(elem):
            if found:
                return False
            found = True
    return found

async def BAL_is_unique(sequence:list, mapping) -> bool:
    mapped = [mapping(elem) for elem in sequence]
    return len(set(mapped)) == len(mapped)

async def BAL_map(sequence:list, mapping) -> list:
    return [mapping(elem) for elem in sequence]

async def BAL_reduce(sequence:list, reduce_fn, aggregator) -> any:
    for elem in sequence:
        aggregator = reduce_fn(aggregator, elem)
    return aggregator


############################################
#
#   Comment functions
#
############################################

@app.get("/comment/", response_model=None, tags=["Comment"])
def get_all_comment(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    from sqlalchemy.orm import joinedload

    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Comment)
        query = query.options(joinedload(Comment.commenter))
        query = query.options(joinedload(Comment.post))
        comment_list = query.all()

        # Serialize with relationships included
        result = []
        for comment_item in comment_list:
            item_dict = comment_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if comment_item.commenter:
                related_obj = comment_item.commenter
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['commenter'] = related_dict
            else:
                item_dict['commenter'] = None
            if comment_item.post:
                related_obj = comment_item.post
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['post'] = related_dict
            else:
                item_dict['post'] = None


            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Comment).all()


@app.get("/comment/count/", response_model=None, tags=["Comment"])
def get_count_comment(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Comment entities"""
    count = database.query(Comment).count()
    return {"count": count}


@app.get("/comment/paginated/", response_model=None, tags=["Comment"])
def get_paginated_comment(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Comment entities"""
    total = database.query(Comment).count()
    comment_list = database.query(Comment).offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": comment_list
    }


@app.get("/comment/search/", response_model=None, tags=["Comment"])
def search_comment(
    database: Session = Depends(get_db)
) -> list:
    """Search Comment entities by attributes"""
    query = database.query(Comment)


    results = query.all()
    return results


@app.get("/comment/{comment_id}/", response_model=None, tags=["Comment"])
async def get_comment(comment_id: int, database: Session = Depends(get_db)) -> Comment:
    db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
    if db_comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")

    response_data = {
        "comment": db_comment,
}
    return response_data



@app.post("/comment/", response_model=None, tags=["Comment"])
async def create_comment(comment_data: CommentCreate, database: Session = Depends(get_db)) -> Comment:

    if comment_data.commenter is not None:
        db_commenter = database.query(User).filter(User.id == comment_data.commenter).first()
        if not db_commenter:
            raise HTTPException(status_code=400, detail="User not found")
    else:
        raise HTTPException(status_code=400, detail="User ID is required")
    if comment_data.post is not None:
        db_post = database.query(Post).filter(Post.id == comment_data.post).first()
        if not db_post:
            raise HTTPException(status_code=400, detail="Post not found")
    else:
        raise HTTPException(status_code=400, detail="Post ID is required")

    db_comment = Comment(
        text=comment_data.text,        created_at=comment_data.created_at,        id=comment_data.id,        commenter_id=comment_data.commenter,        post_id=comment_data.post        )

    database.add(db_comment)
    database.commit()
    database.refresh(db_comment)




    return db_comment


@app.post("/comment/bulk/", response_model=None, tags=["Comment"])
async def bulk_create_comment(items: list[CommentCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Comment entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.commenter:
                raise ValueError("User ID is required")
            if not item_data.post:
                raise ValueError("Post ID is required")

            db_comment = Comment(
                text=item_data.text,                created_at=item_data.created_at,                id=item_data.id,                commenter_id=item_data.commenter,                post_id=item_data.post            )
            database.add(db_comment)
            database.flush()  # Get ID without committing
            created_items.append(db_comment.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Comment entities"
    }


@app.delete("/comment/bulk/", response_model=None, tags=["Comment"])
async def bulk_delete_comment(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Comment entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_comment = database.query(Comment).filter(Comment.id == item_id).first()
        if db_comment:
            database.delete(db_comment)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Comment entities"
    }

@app.put("/comment/{comment_id}/", response_model=None, tags=["Comment"])
async def update_comment(comment_id: int, comment_data: CommentCreate, database: Session = Depends(get_db)) -> Comment:
    db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
    if db_comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")

    setattr(db_comment, 'text', comment_data.text)
    setattr(db_comment, 'created_at', comment_data.created_at)
    setattr(db_comment, 'id', comment_data.id)
    if comment_data.commenter is not None:
        db_commenter = database.query(User).filter(User.id == comment_data.commenter).first()
        if not db_commenter:
            raise HTTPException(status_code=400, detail="User not found")
        setattr(db_comment, 'commenter_id', comment_data.commenter)
    if comment_data.post is not None:
        db_post = database.query(Post).filter(Post.id == comment_data.post).first()
        if not db_post:
            raise HTTPException(status_code=400, detail="Post not found")
        setattr(db_comment, 'post_id', comment_data.post)
    database.commit()
    database.refresh(db_comment)

    return db_comment


@app.delete("/comment/{comment_id}/", response_model=None, tags=["Comment"])
async def delete_comment(comment_id: int, database: Session = Depends(get_db)):
    db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
    if db_comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")
    database.delete(db_comment)
    database.commit()
    return db_comment





############################################
#
#   Post functions
#
############################################

@app.get("/post/", response_model=None, tags=["Post"])
def get_all_post(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    from sqlalchemy.orm import joinedload

    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Post)
        query = query.options(joinedload(Post.author))
        post_list = query.all()

        # Serialize with relationships included
        result = []
        for post_item in post_list:
            item_dict = post_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if post_item.author:
                related_obj = post_item.author
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['author'] = related_dict
            else:
                item_dict['author'] = None

            # Add many-to-many and one-to-many relationship objects (full details)
            comment_list = database.query(Comment).filter(Comment.post_id == post_item.id).all()
            item_dict['comments'] = []
            for comment_obj in comment_list:
                comment_dict = comment_obj.__dict__.copy()
                comment_dict.pop('_sa_instance_state', None)
                item_dict['comments'].append(comment_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Post).all()


@app.get("/post/count/", response_model=None, tags=["Post"])
def get_count_post(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Post entities"""
    count = database.query(Post).count()
    return {"count": count}


@app.get("/post/paginated/", response_model=None, tags=["Post"])
def get_paginated_post(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Post entities"""
    total = database.query(Post).count()
    post_list = database.query(Post).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": post_list
        }

    result = []
    for post_item in post_list:
        comments_ids = database.query(Comment.id).filter(Comment.post_id == post_item.id).all()
        item_data = {
            "post": post_item,
            "comments_ids": [x[0] for x in comments_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }


@app.get("/post/search/", response_model=None, tags=["Post"])
def search_post(
    database: Session = Depends(get_db)
) -> list:
    """Search Post entities by attributes"""
    query = database.query(Post)


    results = query.all()
    return results


@app.get("/post/{post_id}/", response_model=None, tags=["Post"])
async def get_post(post_id: int, database: Session = Depends(get_db)) -> Post:
    db_post = database.query(Post).filter(Post.id == post_id).first()
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    comments_ids = database.query(Comment.id).filter(Comment.post_id == db_post.id).all()
    response_data = {
        "post": db_post,
        "comments_ids": [x[0] for x in comments_ids]}
    return response_data



@app.post("/post/", response_model=None, tags=["Post"])
async def create_post(post_data: PostCreate, database: Session = Depends(get_db)) -> Post:

    if post_data.author is not None:
        db_author = database.query(User).filter(User.id == post_data.author).first()
        if not db_author:
            raise HTTPException(status_code=400, detail="User not found")
    else:
        raise HTTPException(status_code=400, detail="User ID is required")

    db_post = Post(
        created_at=post_data.created_at,        content=post_data.content,        title=post_data.title,        id=post_data.id,        author_id=post_data.author        )

    database.add(db_post)
    database.commit()
    database.refresh(db_post)

    if post_data.comments:
        # Validate that all Comment IDs exist
        for comment_id in post_data.comments:
            db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
            if not db_comment:
                raise HTTPException(status_code=400, detail=f"Comment with id {comment_id} not found")

        # Update the related entities with the new foreign key
        database.query(Comment).filter(Comment.id.in_(post_data.comments)).update(
            {Comment.post_id: db_post.id}, synchronize_session=False
        )
        database.commit()



    comments_ids = database.query(Comment.id).filter(Comment.post_id == db_post.id).all()
    response_data = {
        "post": db_post,
        "comments_ids": [x[0] for x in comments_ids]    }
    return response_data


@app.post("/post/bulk/", response_model=None, tags=["Post"])
async def bulk_create_post(items: list[PostCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Post entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.author:
                raise ValueError("User ID is required")

            db_post = Post(
                created_at=item_data.created_at,                content=item_data.content,                title=item_data.title,                id=item_data.id,                author_id=item_data.author            )
            database.add(db_post)
            database.flush()  # Get ID without committing
            created_items.append(db_post.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Post entities"
    }


@app.delete("/post/bulk/", response_model=None, tags=["Post"])
async def bulk_delete_post(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Post entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_post = database.query(Post).filter(Post.id == item_id).first()
        if db_post:
            database.delete(db_post)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Post entities"
    }

@app.put("/post/{post_id}/", response_model=None, tags=["Post"])
async def update_post(post_id: int, post_data: PostCreate, database: Session = Depends(get_db)) -> Post:
    db_post = database.query(Post).filter(Post.id == post_id).first()
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    setattr(db_post, 'created_at', post_data.created_at)
    setattr(db_post, 'content', post_data.content)
    setattr(db_post, 'title', post_data.title)
    setattr(db_post, 'id', post_data.id)
    if post_data.author is not None:
        db_author = database.query(User).filter(User.id == post_data.author).first()
        if not db_author:
            raise HTTPException(status_code=400, detail="User not found")
        setattr(db_post, 'author_id', post_data.author)
    if post_data.comments is not None:
        # Clear all existing relationships (set foreign key to NULL)
        database.query(Comment).filter(Comment.post_id == db_post.id).update(
            {Comment.post_id: None}, synchronize_session=False
        )

        # Set new relationships if list is not empty
        if post_data.comments:
            # Validate that all IDs exist
            for comment_id in post_data.comments:
                db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
                if not db_comment:
                    raise HTTPException(status_code=400, detail=f"Comment with id {comment_id} not found")

            # Update the related entities with the new foreign key
            database.query(Comment).filter(Comment.id.in_(post_data.comments)).update(
                {Comment.post_id: db_post.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_post)

    comments_ids = database.query(Comment.id).filter(Comment.post_id == db_post.id).all()
    response_data = {
        "post": db_post,
        "comments_ids": [x[0] for x in comments_ids]    }
    return response_data


@app.delete("/post/{post_id}/", response_model=None, tags=["Post"])
async def delete_post(post_id: int, database: Session = Depends(get_db)):
    db_post = database.query(Post).filter(Post.id == post_id).first()
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    database.delete(db_post)
    database.commit()
    return db_post





############################################
#
#   User functions
#
############################################

@app.get("/user/", response_model=None, tags=["User"])
def get_all_user(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    from sqlalchemy.orm import joinedload

    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(User)
        user_list = query.all()

        # Serialize with relationships included
        result = []
        for user_item in user_list:
            item_dict = user_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)

            # Add many-to-many and one-to-many relationship objects (full details)
            comment_list = database.query(Comment).filter(Comment.commenter_id == user_item.id).all()
            item_dict['comments'] = []
            for comment_obj in comment_list:
                comment_dict = comment_obj.__dict__.copy()
                comment_dict.pop('_sa_instance_state', None)
                item_dict['comments'].append(comment_dict)
            post_list = database.query(Post).filter(Post.author_id == user_item.id).all()
            item_dict['posts'] = []
            for post_obj in post_list:
                post_dict = post_obj.__dict__.copy()
                post_dict.pop('_sa_instance_state', None)
                item_dict['posts'].append(post_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(User).all()


@app.get("/user/count/", response_model=None, tags=["User"])
def get_count_user(database: Session = Depends(get_db)) -> dict:
    """Get the total count of User entities"""
    count = database.query(User).count()
    return {"count": count}


@app.get("/user/paginated/", response_model=None, tags=["User"])
def get_paginated_user(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of User entities"""
    total = database.query(User).count()
    user_list = database.query(User).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": user_list
        }

    result = []
    for user_item in user_list:
        comments_ids = database.query(Comment.id).filter(Comment.commenter_id == user_item.id).all()
        posts_ids = database.query(Post.id).filter(Post.author_id == user_item.id).all()
        item_data = {
            "user": user_item,
            "comments_ids": [x[0] for x in comments_ids],            "posts_ids": [x[0] for x in posts_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }


@app.get("/user/search/", response_model=None, tags=["User"])
def search_user(
    database: Session = Depends(get_db)
) -> list:
    """Search User entities by attributes"""
    query = database.query(User)


    results = query.all()
    return results


@app.get("/user/{user_id}/", response_model=None, tags=["User"])
async def get_user(user_id: int, database: Session = Depends(get_db)) -> User:
    db_user = database.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    comments_ids = database.query(Comment.id).filter(Comment.commenter_id == db_user.id).all()
    posts_ids = database.query(Post.id).filter(Post.author_id == db_user.id).all()
    response_data = {
        "user": db_user,
        "comments_ids": [x[0] for x in comments_ids],        "posts_ids": [x[0] for x in posts_ids]}
    return response_data



@app.post("/user/", response_model=None, tags=["User"])
async def create_user(user_data: UserCreate, database: Session = Depends(get_db)) -> User:


    db_user = User(
        bio=user_data.bio,        id=user_data.id,        email=user_data.email,        is_active=user_data.is_active,        username=user_data.username,        password_hash=user_data.password_hash        )

    database.add(db_user)
    database.commit()
    database.refresh(db_user)

    if user_data.comments:
        # Validate that all Comment IDs exist
        for comment_id in user_data.comments:
            db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
            if not db_comment:
                raise HTTPException(status_code=400, detail=f"Comment with id {comment_id} not found")

        # Update the related entities with the new foreign key
        database.query(Comment).filter(Comment.id.in_(user_data.comments)).update(
            {Comment.commenter_id: db_user.id}, synchronize_session=False
        )
        database.commit()
    if user_data.posts:
        # Validate that all Post IDs exist
        for post_id in user_data.posts:
            db_post = database.query(Post).filter(Post.id == post_id).first()
            if not db_post:
                raise HTTPException(status_code=400, detail=f"Post with id {post_id} not found")

        # Update the related entities with the new foreign key
        database.query(Post).filter(Post.id.in_(user_data.posts)).update(
            {Post.author_id: db_user.id}, synchronize_session=False
        )
        database.commit()



    comments_ids = database.query(Comment.id).filter(Comment.commenter_id == db_user.id).all()
    posts_ids = database.query(Post.id).filter(Post.author_id == db_user.id).all()
    response_data = {
        "user": db_user,
        "comments_ids": [x[0] for x in comments_ids],        "posts_ids": [x[0] for x in posts_ids]    }
    return response_data


@app.post("/user/bulk/", response_model=None, tags=["User"])
async def bulk_create_user(items: list[UserCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple User entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item

            db_user = User(
                bio=item_data.bio,                id=item_data.id,                email=item_data.email,                is_active=item_data.is_active,                username=item_data.username,                password_hash=item_data.password_hash            )
            database.add(db_user)
            database.flush()  # Get ID without committing
            created_items.append(db_user.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} User entities"
    }


@app.delete("/user/bulk/", response_model=None, tags=["User"])
async def bulk_delete_user(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple User entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_user = database.query(User).filter(User.id == item_id).first()
        if db_user:
            database.delete(db_user)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} User entities"
    }

@app.put("/user/{user_id}/", response_model=None, tags=["User"])
async def update_user(user_id: int, user_data: UserCreate, database: Session = Depends(get_db)) -> User:
    db_user = database.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    setattr(db_user, 'bio', user_data.bio)
    setattr(db_user, 'id', user_data.id)
    setattr(db_user, 'email', user_data.email)
    setattr(db_user, 'is_active', user_data.is_active)
    setattr(db_user, 'username', user_data.username)
    setattr(db_user, 'password_hash', user_data.password_hash)
    if user_data.comments is not None:
        # Clear all existing relationships (set foreign key to NULL)
        database.query(Comment).filter(Comment.commenter_id == db_user.id).update(
            {Comment.commenter_id: None}, synchronize_session=False
        )

        # Set new relationships if list is not empty
        if user_data.comments:
            # Validate that all IDs exist
            for comment_id in user_data.comments:
                db_comment = database.query(Comment).filter(Comment.id == comment_id).first()
                if not db_comment:
                    raise HTTPException(status_code=400, detail=f"Comment with id {comment_id} not found")

            # Update the related entities with the new foreign key
            database.query(Comment).filter(Comment.id.in_(user_data.comments)).update(
                {Comment.commenter_id: db_user.id}, synchronize_session=False
            )
    if user_data.posts is not None:
        # Clear all existing relationships (set foreign key to NULL)
        database.query(Post).filter(Post.author_id == db_user.id).update(
            {Post.author_id: None}, synchronize_session=False
        )

        # Set new relationships if list is not empty
        if user_data.posts:
            # Validate that all IDs exist
            for post_id in user_data.posts:
                db_post = database.query(Post).filter(Post.id == post_id).first()
                if not db_post:
                    raise HTTPException(status_code=400, detail=f"Post with id {post_id} not found")

            # Update the related entities with the new foreign key
            database.query(Post).filter(Post.id.in_(user_data.posts)).update(
                {Post.author_id: db_user.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_user)

    comments_ids = database.query(Comment.id).filter(Comment.commenter_id == db_user.id).all()
    posts_ids = database.query(Post.id).filter(Post.author_id == db_user.id).all()
    response_data = {
        "user": db_user,
        "comments_ids": [x[0] for x in comments_ids],        "posts_ids": [x[0] for x in posts_ids]    }
    return response_data


@app.delete("/user/{user_id}/", response_model=None, tags=["User"])
async def delete_user(user_id: int, database: Session = Depends(get_db)):
    db_user = database.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    database.delete(db_user)
    database.commit()
    return db_user







############################################
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



