CREATE TABLE author (
	id INTEGER NOT NULL,
	name TEXT NOT NULL,
	email TEXT NOT NULL,
	PRIMARY KEY (id)
);

CREATE TABLE library (
	id INTEGER NOT NULL,
	name TEXT NOT NULL,
	address TEXT NOT NULL,
	PRIMARY KEY (id)
);

CREATE TABLE book (
	id INTEGER NOT NULL,
	title TEXT NOT NULL,
	pages INTEGER NOT NULL,
	release DATE NOT NULL,
	locatedIn_id INTEGER NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(locatedIn_id) REFERENCES library (id)
);

CREATE TABLE book_author_assoc (
	publishes INTEGER NOT NULL,
	writtenBy INTEGER NOT NULL,
	PRIMARY KEY (publishes, writtenBy),
	FOREIGN KEY(publishes) REFERENCES book (id),
	FOREIGN KEY(writtenBy) REFERENCES author (id)
);
