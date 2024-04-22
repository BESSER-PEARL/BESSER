CREATE TABLE IF NOT EXISTS Library (
    Library_id SERIAL PRIMARY KEY,
    address varchar(100),      
    name varchar(100)
);

CREATE TABLE IF NOT EXISTS Book (
    Book_id SERIAL PRIMARY KEY,
    title varchar(100),      
    pages int,      
    release timestamp
);

CREATE TABLE IF NOT EXISTS Author (
    Author_id SERIAL PRIMARY KEY,
    name varchar(100),      
    email varchar(100)
);

CREATE TABLE IF NOT EXISTS Author_Book (
    Author_id INT REFERENCES Author(Author_id),
    Book_id INT REFERENCES Book(Book_id),
    PRIMARY KEY (Author_id, Book_id)
);

ALTER TABLE Book
ADD COLUMN Library_id INT REFERENCES Library(Library_id);

