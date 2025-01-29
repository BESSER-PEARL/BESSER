import os, json
from fastapi import FastAPI, HTTPException
from pydantic_classes import *

app = FastAPI()

############################################
#
# Lists to store the data (json)
#
############################################

book_list = []
author_list = []
library_list = []


############################################
#
#   Book functions
#
############################################
@app.get("/book/", response_model=List[Book], tags=["book"])
def get_book():
    return book_list

@app.get("/book/{attribute_id}/", response_model=Book, tags=["book"])
def get_book(attribute_id : str):   
    for book in book_list:
        if book.id_to_change== attribute_id:
            return book
    raise HTTPException(status_code=404, detail="Book not found")

@app.post("/book/", response_model=Book, tags=["book"])
def create_book(book: Book):
    book_list.append(book)
    return book

@app.put("/book/{attribute_id}/", response_model=Book, tags=["book"]) 
def change_book(attribute_id : str, updated_book: Book): 
    for index, book in enumerate(book_list): 
        if book.id_to_change == attribute_id:
            book_list[index] = updated_book
            return updated_book
    raise HTTPException(status_code=404, detail="Book not found")

@app.patch("/book/{attribute_id}/{attribute_to_change}", response_model=Book, tags=["book"])
def update_book(attribute_id : str,  attribute_to_change: str, updated_data: str):
    for book in book_list:
        if book.id_to_change == attribute_id:
            if hasattr(book, attribute_to_change):
                setattr(book, attribute_to_change, updated_data)
                return book
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Book not found")

@app.delete("/book/{attribute_id}/", tags=["book"])
def delete_book(attribute_id : str):   
    for index, book in enumerate(book_list):
        if book.id_to_change == attribute_id:
            book_list.pop(index)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found") 

############################################
#
#   Author functions
#
############################################
@app.get("/author/", response_model=List[Author], tags=["author"])
def get_author():
    return author_list

@app.get("/author/{attribute_id}/", response_model=Author, tags=["author"])
def get_author(attribute_id : str):   
    for author in author_list:
        if author.id_to_change== attribute_id:
            return author
    raise HTTPException(status_code=404, detail="Author not found")

@app.post("/author/", response_model=Author, tags=["author"])
def create_author(author: Author):
    author_list.append(author)
    return author

@app.put("/author/{attribute_id}/", response_model=Author, tags=["author"]) 
def change_author(attribute_id : str, updated_author: Author): 
    for index, author in enumerate(author_list): 
        if author.id_to_change == attribute_id:
            author_list[index] = updated_author
            return updated_author
    raise HTTPException(status_code=404, detail="Author not found")

@app.patch("/author/{attribute_id}/{attribute_to_change}", response_model=Author, tags=["author"])
def update_author(attribute_id : str,  attribute_to_change: str, updated_data: str):
    for author in author_list:
        if author.id_to_change == attribute_id:
            if hasattr(author, attribute_to_change):
                setattr(author, attribute_to_change, updated_data)
                return author
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Author not found")

@app.delete("/author/{attribute_id}/", tags=["author"])
def delete_author(attribute_id : str):   
    for index, author in enumerate(author_list):
        if author.id_to_change == attribute_id:
            author_list.pop(index)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Author not found") 

############################################
#
#   Library functions
#
############################################
@app.get("/library/", response_model=List[Library], tags=["library"])
def get_library():
    return library_list

@app.get("/library/{attribute_id}/", response_model=Library, tags=["library"])
def get_library(attribute_id : str):   
    for library in library_list:
        if library.id_to_change== attribute_id:
            return library
    raise HTTPException(status_code=404, detail="Library not found")

@app.post("/library/", response_model=Library, tags=["library"])
def create_library(library: Library):
    library_list.append(library)
    return library

@app.put("/library/{attribute_id}/", response_model=Library, tags=["library"]) 
def change_library(attribute_id : str, updated_library: Library): 
    for index, library in enumerate(library_list): 
        if library.id_to_change == attribute_id:
            library_list[index] = updated_library
            return updated_library
    raise HTTPException(status_code=404, detail="Library not found")

@app.patch("/library/{attribute_id}/{attribute_to_change}", response_model=Library, tags=["library"])
def update_library(attribute_id : str,  attribute_to_change: str, updated_data: str):
    for library in library_list:
        if library.id_to_change == attribute_id:
            if hasattr(library, attribute_to_change):
                setattr(library, attribute_to_change, updated_data)
                return library
            else:
                raise HTTPException(status_code=400, detail=f"Attribute '{attribute_to_change}' does not exist")
    raise HTTPException(status_code=404, detail="Library not found")

@app.delete("/library/{attribute_id}/", tags=["library"])
def delete_library(attribute_id : str):   
    for index, library in enumerate(library_list):
        if library.id_to_change == attribute_id:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)



