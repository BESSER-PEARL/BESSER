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

SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


app = FastAPI()

# Initialize database session
def get_db():
    database = SessionLocal()
    yield database
    database.close()

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
    return db_library




@app.post("/library/", response_model=None)
async def create_library(library: LibraryCreate, database: Session = Depends(get_db)) -> Library:


    db_library = Library(name=library.name, address=library.address)

    database.add(db_library)
    database.commit()
    database.refresh(db_library)


    
    return db_library


@app.put("/library/{library_id}/", response_model=None)
async def update_library(library_id: int, library: LibraryCreate, database: Session = Depends(get_db)) -> Library:
    db_library = database.query(Library).filter(Library.id == library_id).first()
    if db_library is None:
        raise HTTPException(status_code=404, detail="Library not found")

    for key, value in library.dict().items():
        setattr(db_library, key, value)

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
    return db_book




@app.post("/book/", response_model=None)
async def create_book(book: BookCreate, database: Session = Depends(get_db)) -> Book:

    if book.library_id:
        db_library = database.query(Library).filter(Library.id == book.library_id).first()
        if not db_library:
            raise HTTPException(status_code=400, detail="Library not found")

    db_book = Book(pages=book.pages, release=book.release, title=book.title, library_id=book.library_id)

    database.add(db_book)
    database.commit()
    database.refresh(db_book)

    if book.authors_id and len(book.authors_id) != []:
        for x in book.authors_id:
            db_author = database.query(Author).filter(Author.id == x).first()
            if not db_author:
                raise HTTPException(status_code=404, detail=f"Author with ID {x} not found")
            # Create the association
            association = book_author_assoc.insert().values(book_id=db_book.id, author_id=db_author.id)
            database.execute(association)
            database.commit()

    
    return db_book


@app.put("/book/{book_id}/", response_model=None)
async def update_book(book_id: int, book: BookCreate, database: Session = Depends(get_db)) -> Book:
    db_book = database.query(Book).filter(Book.id == book_id).first()
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    for key, value in book.dict().items():
        setattr(db_book, key, value)

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
    return db_author




@app.post("/author/", response_model=None)
async def create_author(author: AuthorCreate, database: Session = Depends(get_db)) -> Author:


    db_author = Author(name=author.name, email=author.email)

    database.add(db_author)
    database.commit()
    database.refresh(db_author)

    if author.books_id and len(author.books_id) != []:
        for x in author.books_id:
            db_book = database.query(Book).filter(Book.id == x).first()
            if not db_book:
                raise HTTPException(status_code=404, detail=f"Book with ID {x} not found")
            # Create the association
            association = book_author_assoc.insert().values(author_id=db_author.id, book_id=db_book.id)
            database.execute(association)
            database.commit()

    
    return db_author


@app.put("/author/{author_id}/", response_model=None)
async def update_author(author_id: int, author: AuthorCreate, database: Session = Depends(get_db)) -> Author:
    db_author = database.query(Author).filter(Author.id == author_id).first()
    if db_author is None:
        raise HTTPException(status_code=404, detail="Author not found")

    for key, value in author.dict().items():
        setattr(db_author, key, value)

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
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    openapi_schema = app.openapi()
    output_dir = os.path.join(os.getcwd(), 'output_backend')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'openapi_specs.json')
    print(f"Writing OpenAPI schema to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(openapi_schema, file)
    uvicorn.run(app, host="0.0.0.0", port=8000)



