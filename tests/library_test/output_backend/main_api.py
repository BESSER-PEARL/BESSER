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
#   Author functions
#
############################################
@app.get("/author/", response_model=None)
def get_all_author(database: Session = Depends(get_db)) -> list[Author]:
    author_list = database.query(Author).all()
    return author_list


@app.get("/author/{author_id}/", response_model=None)
async def get_author(author_id : int, database: Session = Depends(get_db)) -> Author:
    author = database.query(Author).filter(Author.id == {author_id}).first()
    if author is None:
        raise HTTPException(status_code=404, detail="Author not found")
    database.delete(author)
    database.commit()
    return author


@app.post("/author/", response_model=None, tags=["author"])
def create_author(author: AuthorCreate, database: Session = Depends(get_db)) -> Author:
    db_author = Author(**author.dict())
    database.add(db_author)
    database.commit()
    database.refresh(db_author)
    return db_author

@app.delete("/author/{author_id}/", response_model=None)
def delete_author(author_id: int, database: Session = Depends(get_db)):
    author = database.query(Author).filter(Author.id == {author_id}).first()
    if author is None:
        raise HTTPException(status_code=404, detail="Author not found")
    database.delete(author)
    database.commit()
    return author



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
async def get_book(book_id : int, database: Session = Depends(get_db)) -> Book:
    book = database.query(Book).filter(Book.id == {book_id}).first()
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    database.delete(book)
    database.commit()
    return book


@app.post("/book/", response_model=None, tags=["book"])
def create_book(book: BookCreate, database: Session = Depends(get_db)) -> Book:
    db_book = Book(**book.dict())
    database.add(db_book)
    database.commit()
    database.refresh(db_book)
    return db_book

@app.delete("/book/{book_id}/", response_model=None)
def delete_book(book_id: int, database: Session = Depends(get_db)):
    book = database.query(Book).filter(Book.id == {book_id}).first()
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    database.delete(book)
    database.commit()
    return book



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
async def get_library(library_id : int, database: Session = Depends(get_db)) -> Library:
    library = database.query(Library).filter(Library.id == {library_id}).first()
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    database.delete(library)
    database.commit()
    return library


@app.post("/library/", response_model=None, tags=["library"])
def create_library(library: LibraryCreate, database: Session = Depends(get_db)) -> Library:
    db_library = Library(**library.dict())
    database.add(db_library)
    database.commit()
    database.refresh(db_library)
    return db_library

@app.delete("/library/{library_id}/", response_model=None)
def delete_library(library_id: int, database: Session = Depends(get_db)):
    library = database.query(Library).filter(Library.id == {library_id}).first()
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    database.delete(library)
    database.commit()
    return library





############################################
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    openapi_schema = app.openapi()
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'openapi_specs.json')
    print(f"Writing OpenAPI schema to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(openapi_schema, file)
    uvicorn.run(app, host="0.0.0.0", port=8000)



