CREATE TABLE IF NOT EXISTS Author (
    author_id SERIAL PRIMARY KEY,
    name varchar(100),      
    email varchar(100)
);

CREATE TABLE IF NOT EXISTS Book (
    book_id SERIAL PRIMARY KEY,
    title varchar(100),      
    pages int,      
    release date
);

CREATE TABLE IF NOT EXISTS Library (
    library_id SERIAL PRIMARY KEY,
    name varchar(100),      
    address varchar(100)
);

ALTER TABLE Book
ADD COLUMN locatedIn INT REFERENCES Library(library_id);

CREATE TABLE IF NOT EXISTS book_author_assoc (
    writtenBy INT REFERENCES Author(author_id),
    publishes INT REFERENCES Book(book_id),
    PRIMARY KEY (writtenBy, publishes)
);

