CREATE TABLE IF NOT EXISTS Book (
    Book_id SERIAL PRIMARY KEY,
    pages int,      
    title varchar(100),      
    release timestamp
);

CREATE TABLE IF NOT EXISTS Library (
    Library_id SERIAL PRIMARY KEY,
    address varchar(100),      
    name varchar(100)
);

CREATE TABLE IF NOT EXISTS Author (
    Author_id SERIAL PRIMARY KEY,
    email varchar(100),      
    name varchar(100)
);

CREATE TABLE IF NOT EXISTS Book_Author (
    Book_id INT REFERENCES Book(Book_id),
    Author_id INT REFERENCES Author(Author_id),
    PRIMARY KEY (Book_id, Author_id)
);

ALTER TABLE Book
ADD COLUMN Library_id INT REFERENCES Library(Library_id);

