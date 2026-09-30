import os, json
from fastapi import FastAPI, HTTPException
from pydantic_classes import *

app = FastAPI()

############################################
#
# Lists to store the data (json)
#
############################################

author_list = []
book_list = []
library_list = []


############################################
#
#   Author functions
#
############################################


@app.get("/author/", response_model=List[Author], tags=["author"])
def get_author():
    return author_list

@app.get("/author/{id}/", response_model=Author, tags=["author"])
def get_author(id : int):
    for author in author_list:
        if author.id== id:
            return author
    raise HTTPException(status_code=404, detail="Author not found")

@app.post("/author/", response_model=Author, tags=["author"])
def create_author(author: Author):
    for existing_author in author_list:
        if existing_author.id == author.id:
            raise HTTPException(status_code=400, detail=f"Author with id {existing_author.id} already exists")


    books_id = getattr(author, 'books_id', None)
    if books_id:
        for id in books_id:
            book_exists = any(book.id == id for book in book_list)
            if not book_exists:
                raise HTTPException(status_code=404, detail=f"Book with ID {id} not found")

    author_list.append(author)
    return author




@app.put("/author/{id}/", response_model=Author, tags=["author"])
def change_author(id : int, updated_author: Author):
    for index, author in enumerate(author_list): 
        if author.id == id:
            author_list[index] = updated_author
            return updated_author
    raise HTTPException(status_code=404, detail="Author not found")

@app.patch("/author/{id}/{attribute_to_change}", response_model=Author, tags=["author"])
def update_author(id : int,  attribute_to_change: str, updated_data: str):
    for author in author_list:
        if author.id == id:
            if hasattr(author, attribute_to_change):
                setattr(author, attribute_to_change, updated_data)
                return author
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Author not found")

@app.delete("/author/{id}/", tags=["author"])
def delete_author(id : int):
    for index, author in enumerate(author_list):
        if author.id == id:
            author_list.pop(index)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Author not found") 

############################################
#
#   Book functions
#
############################################


@app.get("/book/", response_model=List[Book], tags=["book"])
def get_book():
    return book_list

@app.get("/book/{id}/", response_model=Book, tags=["book"])
def get_book(id : int):
    for book in book_list:
        if book.id== id:
            return book
    raise HTTPException(status_code=404, detail="Book not found")

@app.post("/book/", response_model=Book, tags=["book"])
def create_book(book: Book):
    for existing_book in book_list:
        if existing_book.id == book.id:
            raise HTTPException(status_code=400, detail=f"Book with id {existing_book.id} already exists")

    library_id = getattr(book, 'library_id', None)
    if library_id is not None:
        library_exists = any(library.id == library_id for library in library_list)
        if not library_exists:
            raise HTTPException(status_code=400, detail="Library not found")

    authors_id = getattr(book, 'authors_id', None)
    if authors_id:
        for id in authors_id:
            author_exists = any(author.id == id for author in author_list)
            if not author_exists:
                raise HTTPException(status_code=404, detail=f"Author with ID {id} not found")

    book_list.append(book)
    return book




@app.put("/book/{id}/", response_model=Book, tags=["book"])
def change_book(id : int, updated_book: Book):
    for index, book in enumerate(book_list): 
        if book.id == id:
            book_list[index] = updated_book
            return updated_book
    raise HTTPException(status_code=404, detail="Book not found")

@app.patch("/book/{id}/{attribute_to_change}", response_model=Book, tags=["book"])
def update_book(id : int,  attribute_to_change: str, updated_data: str):
    for book in book_list:
        if book.id == id:
            if hasattr(book, attribute_to_change):
                setattr(book, attribute_to_change, updated_data)
                return book
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Book not found")

@app.delete("/book/{id}/", tags=["book"])
def delete_book(id : int):
    for index, book in enumerate(book_list):
        if book.id == id:
            book_list.pop(index)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found") 

############################################
#
#   Library functions
#
############################################


@app.get("/library/", response_model=List[Library], tags=["library"])
def get_library():
    return library_list

@app.get("/library/{id}/", response_model=Library, tags=["library"])
def get_library(id : int):
    for library in library_list:
        if library.id== id:
            return library
    raise HTTPException(status_code=404, detail="Library not found")

@app.post("/library/", response_model=Library, tags=["library"])
def create_library(library: Library):
    for existing_library in library_list:
        if existing_library.id == library.id:
            raise HTTPException(status_code=400, detail=f"Library with id {existing_library.id} already exists")



    library_list.append(library)
    return library




@app.put("/library/{id}/", response_model=Library, tags=["library"])
def change_library(id : int, updated_library: Library):
    for index, library in enumerate(library_list): 
        if library.id == id:
            library_list[index] = updated_library
            return updated_library
    raise HTTPException(status_code=404, detail="Library not found")

@app.patch("/library/{id}/{attribute_to_change}", response_model=Library, tags=["library"])
def update_library(id : int,  attribute_to_change: str, updated_data: str):
    for library in library_list:
        if library.id == id:
            if hasattr(library, attribute_to_change):
                setattr(library, attribute_to_change, updated_data)
                return library
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Library not found")

@app.delete("/library/{id}/", tags=["library"])
def delete_library(id : int):
    for index, library in enumerate(library_list):
        if library.id == id:
            library_list.pop(index)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Library not found") 



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
    uvicorn.run(app, host="127.0.0.1", port=8000)



