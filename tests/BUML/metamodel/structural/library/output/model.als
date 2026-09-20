open util/boolean
open util/integer
open util/ordering[Date]

sig Str {}

sig Date {}

one sig d0 extends Date {}
one sig d1 extends Date {}
one sig d2 extends Date {}
one sig d3 extends Date {}
one sig d4 extends Date {}

fact DateOrder {
    d0 = first
    d0.next = d1
    d1.next = d2
    d2.next = d3
    d3.next = d4
    d4 = last
}

sig Author {    
    author_email: Str,
    author_name: Str,
    author_publishes: set Book
}

sig Book {    
    book_pages: Int,
    book_release: Date,
    book_title: Str,
    book_locatedIn: one Library,
    book_writtenBy: set Author
}

sig Library {    
    library_address: Str,
    library_name: Str,
    library_has: set Book
}

fact {
    library_has = ~book_locatedIn
}

fact {
    all b: Book | #(b.book_writtenBy)>=1 
}

fact {
    author_publishes = ~book_writtenBy
}

pred find_instance[] {
 some Author
 some Book
 some Library
 some Str
 some Date
}

run find_instance for 5 Author, 5 Book, 5 Library, 5 Str, 5 int, 5 Date
