import uvicorn
import os, json
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from pydantic_classes import *
from sql_alchemy import *

############################################
#
#   Initialize the database
#
############################################

def init_db():
    SQLALCHEMY_DATABASE_URL = "sqlite:///./Library_model.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal

app = FastAPI()

# Initialize database session
SessionLocal = init_db()
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

############################################
#
#   Author functions
#
############################################
 
 

@app.get("/author/", response_model=None)
def get_all_author(database: Session = Depends(get_db)) -> list[Author]:
    author_list = database.query(Author).all()
    return author_list


@app.get("/author/{author_id}/", response_model=None)
async def get_author(author_id: int, database: Session = Depends(get_db)) -> Author:
    db_author = database.query(Author).filter(Author.id == author_id).first()
    if db_author is None:
        raise HTTPException(status_code=404, detail="Author not found")

    book_ids = database.query(book_author_assoc.c.publishes).filter(book_author_assoc.c.writtenby == db_author.id).all()
    response_data = {
        "author": db_author,
        "book_ids": [x[0] for x in book_ids]}
    return response_data



@app.post("/author/", response_model=None)
async def create_author(author_data: AuthorCreate, database: Session = Depends(get_db)) -> Author:


    db_author = Author(
        name=author_data.name,         email=author_data.email        )

    database.add(db_author)
    database.commit()
    database.refresh(db_author)

    if author_data.publishes:
        for id in author_data.publishes:
            db_book = database.query(Book).filter(Book.id == id).first()
            if not db_book:
                raise HTTPException(status_code=404, detail=f"Book with ID {id} not found")
            # Create the association
            association = book_author_assoc.insert().values(writtenBy=db_author.id, publishes=db_book.id)
            database.execute(association)
            database.commit()

    
    return db_author


@app.put("/author/{author_id}/", response_model=None)
async def update_author(author_id: int, author_data: AuthorCreate, database: Session = Depends(get_db)) -> Author:
    db_author = database.query(Author).filter(Author.id == author_id).first()
    if db_author is None:
        raise HTTPException(status_code=404, detail="Author not found")

    setattr(db_author, 'name', author_data.name)
    setattr(db_author, 'email', author_data.email)
    existing_book_ids = [assoc.publishes for assoc in database.execute(
        book_author_assoc.select().where(book_author_assoc.c.writtenby == db_author.id))]
    
    books_to_remove = set(existing_book_ids) - set(author_data.publishes)
    for book_id in books_to_remove:
        association = book_author_assoc.delete().where(
            book_author_assoc.c.writtenby == db_author.id and book_author_assoc.c.book_id == book_id)
        database.execute(association)

    new_book_ids = set(author_data.publishes) - set(existing_book_ids)
    for book_id in new_book_ids:
        db_book = database.query(Book).filter(Author.id == author_id).first()
        if db_author is None:
            raise HTTPException(status_code=404, detail="Author with ID author_id not found")
        association = book_author_assoc.insert().values(publishes=db_book.id, writtenby=db_author.id)
        database.execute(association)
    database.commit()
    database.refresh(db_author)
    return db_author


@app.delete("/author/{author_id}/", response_model=None)
async def delete_author(author_id: int, database: Session = Depends(get_db)):
    db_author = database.query(Author).filter(Author.id == author_id).first()
    if db_author is None:
        raise HTTPException(status_code=404, detail="Author not found")
    database.delete(db_author)
    database.commit()
    return db_author



############################################
#
#   Book functions
#
############################################
 
 
 
 

@app.get("/book/", response_model=None)
def get_all_book(database: Session = Depends(get_db)) -> list[Book]:
    book_list = database.query(Book).all()
    return book_list


@app.get("/book/{book_id}/", response_model=None)
async def get_book(book_id: int, database: Session = Depends(get_db)) -> Book:
    db_book = database.query(Book).filter(Book.id == book_id).first()
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    author_ids = database.query(book_author_assoc.c.writtenby).filter(book_author_assoc.c.publishes == db_book.id).all()
    response_data = {
        "book": db_book,
        "author_ids": [x[0] for x in author_ids]}
    return response_data



@app.post("/book/", response_model=None)
async def create_book(book_data: BookCreate, database: Session = Depends(get_db)) -> Book:

    if book_data.locatedIn is not None:
        db_locatedIn = database.query(Library).filter(Library.id == book_data.locatedIn).first()
        if not db_locatedIn:
            raise HTTPException(status_code=400, detail="Library not found")
    else:
        raise HTTPException(status_code=400, detail="Library ID is required")

    db_book = Book(
        title=book_data.title,         pages=book_data.pages,         release=book_data.release, locatedIn=db_locatedIn        )

    database.add(db_book)
    database.commit()
    database.refresh(db_book)

    if book_data.writtenBy:
        for id in book_data.writtenBy:
            db_author = database.query(Author).filter(Author.id == id).first()
            if not db_author:
                raise HTTPException(status_code=404, detail=f"Author with ID {id} not found")
            # Create the association
            association = book_author_assoc.insert().values(publishes=db_book.id, writtenBy=db_author.id)
            database.execute(association)
            database.commit()

    
    return db_book


@app.put("/book/{book_id}/", response_model=None)
async def update_book(book_id: int, book_data: BookCreate, database: Session = Depends(get_db)) -> Book:
    db_book = database.query(Book).filter(Book.id == book_id).first()
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    setattr(db_book, 'title', book_data.title)
    setattr(db_book, 'pages', book_data.pages)
    setattr(db_book, 'release', book_data.release)
    existing_author_ids = [assoc.writtenby for assoc in database.execute(
        book_author_assoc.select().where(book_author_assoc.c.publishes == db_book.id))]
    
    authors_to_remove = set(existing_author_ids) - set(book_data.writtenBy)
    for author_id in authors_to_remove:
        association = book_author_assoc.delete().where(
            book_author_assoc.c.publishes == db_book.id and book_author_assoc.c.author_id == author_id)
        database.execute(association)

    new_author_ids = set(book_data.writtenBy) - set(existing_author_ids)
    for author_id in new_author_ids:
        db_author = database.query(Author).filter(Book.id == book_id).first()
        if db_book is None:
            raise HTTPException(status_code=404, detail="Book with ID book_id not found")
        association = book_author_assoc.insert().values(writtenby=db_author.id, publishes=db_book.id)
        database.execute(association)
    database.commit()
    database.refresh(db_book)
    return db_book


@app.delete("/book/{book_id}/", response_model=None)
async def delete_book(book_id: int, database: Session = Depends(get_db)):
    db_book = database.query(Book).filter(Book.id == book_id).first()
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    database.delete(db_book)
    database.commit()
    return db_book



############################################
#
#   Library functions
#
############################################
 
 

@app.get("/library/", response_model=None)
def get_all_library(database: Session = Depends(get_db)) -> list[Library]:
    library_list = database.query(Library).all()
    return library_list


@app.get("/library/{library_id}/", response_model=None)
async def get_library(library_id: int, database: Session = Depends(get_db)) -> Library:
    db_library = database.query(Library).filter(Library.id == library_id).first()
    if db_library is None:
        raise HTTPException(status_code=404, detail="Library not found")

    response_data = {
        "library": db_library,
}
    return response_data



@app.post("/library/", response_model=None)
async def create_library(library_data: LibraryCreate, database: Session = Depends(get_db)) -> Library:


    db_library = Library(
        address=library_data.address,         name=library_data.name        )

    database.add(db_library)
    database.commit()
    database.refresh(db_library)


    
    return db_library


@app.put("/library/{library_id}/", response_model=None)
async def update_library(library_id: int, library_data: LibraryCreate, database: Session = Depends(get_db)) -> Library:
    db_library = database.query(Library).filter(Library.id == library_id).first()
    if db_library is None:
        raise HTTPException(status_code=404, detail="Library not found")

    setattr(db_library, 'address', library_data.address)
    setattr(db_library, 'name', library_data.name)
    database.commit()
    database.refresh(db_library)
    return db_library


@app.delete("/library/{library_id}/", response_model=None)
async def delete_library(library_id: int, database: Session = Depends(get_db)):
    db_library = database.query(Library).filter(Library.id == library_id).first()
    if db_library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    database.delete(db_library)
    database.commit()
    return db_library





############################################
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    openapi_schema = app.openapi()
    output_dir = os.path.join(os.getcwd(), 'output_backend')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'openapi_specs.json')
    print(f"Writing OpenAPI schema to {output_file}")
    print("Swagger UI available at 0.0.0.0:8000/docs")
    with open(output_file, 'w') as file:
        json.dump(openapi_schema, file)
    uvicorn.run(app, host="0.0.0.0", port= 8000)



