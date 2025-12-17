# BESSER API - Method Endpoints Testing Guide

This document contains curl commands to test all the generated method endpoints for Book and Library entities.

## Prerequisites

1. Start the API server:
```bash
cd C:\Users\sulejmani\Desktop\BESSER\output_backend
python main_api.py
```

2. The API will be available at `http://localhost:8000`

---

## BOOK METHODS

### 1. Instance Methods (requires book_id)

#### 1.1 hello_world() - Simple greeting method
```bash
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/hello_world/" -Method POST -ContentType "application/json" -Body "{}"
```

Expected Response:
```json
{
  "book_id": 1,
  "method": "hello_world",
  "status": "executed",
  "result": "The Book Title",
  "output": "Hello from The Book Title!\n"
}
```

---

#### 1.2 calculate_discount(discount) - Calculate pages after discount
```bash
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/calculate_discount/" -Method POST -ContentType "application/json" -Body '{"discount": 0.2}'
```

Expected Response (for a book with 100 pages and 20% discount):
```json
{
  "book_id": 1,
  "method": "calculate_discount",
  "status": "executed",
  "result": "80",
  "output": "Original pages: 100, After 20.0% discount: 80\n"
}
```

Test with different discount rates:
```bash
# 10% discount
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/calculate_discount/" -Method POST -ContentType "application/json" -Body '{"discount": 0.1}'

# 50% discount
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/calculate_discount/" -Method POST -ContentType "application/json" -Body '{"discount": 0.5}'
```

---

#### 1.3 is_long(min_pages) - Check if book exceeds minimum page count
```bash
# Check if book has more than 300 pages (default)
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_long/" -Method POST -ContentType "application/json" -Body '{"min_pages": 300}'

# Check if book has more than 100 pages
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_long/" -Method POST -ContentType "application/json" -Body '{"min_pages": 100}'

# Check if book has more than 500 pages
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_long/" -Method POST -ContentType "application/json" -Body '{"min_pages": 500}'
```

Expected Response:
```json
{
  "book_id": 1,
  "method": "is_long",
  "status": "executed",
  "result": "True",
  "output": "Book 'The Book Title' has 350 pages. Is long (300+): True\n"
}
```

---

#### 1.4 is_old(years) - Check if book was published N years ago
```bash
# Check if book is older than 10 years (default)
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_old/" -Method POST -ContentType "application/json" -Body '{"years": 10}'

# Check if book is older than 5 years
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_old/" -Method POST -ContentType "application/json" -Body '{"years": 5}'

# Check if book is older than 20 years
Invoke-WebRequest -Uri "http://localhost:8000/book/1/methods/is_old/" -Method POST -ContentType "application/json" -Body '{"years": 20}'
```

Expected Response:
```json
{
  "book_id": 1,
  "method": "is_old",
  "status": "executed",
  "result": "False",
  "output": "Book 'The Book Title' published 2 years ago. Is old (10+ years): False\n"
}
```

---

### 2. Class-Level Methods (no book_id required)

#### 2.1 get_longest() - Get the book with the most pages
```bash
Invoke-WebRequest -Uri "http://localhost:8000/book/methods/get_longest/" -Method POST -ContentType "application/json" -Body '{}'
```

Expected Response:
```json
{
  "class": "Book",
  "method": "get_longest",
  "status": "executed",
  "result": {
    "id": 2,
    "pages": 500,
    "title": "Long Novel",
    "release": "2020-06-15"
  },
  "output": null
}
```

---

#### 2.2 count_long_books(min_pages) - Count books with minimum page count
```bash
# Count books with more than 300 pages
Invoke-WebRequest -Uri "http://localhost:8000/book/methods/count_long_books/" -Method POST -ContentType "application/json" -Body '{"min_pages": 300}'

# Count books with more than 100 pages
Invoke-WebRequest -Uri "http://localhost:8000/book/methods/count_long_books/" -Method POST -ContentType "application/json" -Body '{"min_pages": 100}'

# Count books with more than 500 pages
Invoke-WebRequest -Uri "http://localhost:8000/book/methods/count_long_books/" -Method POST -ContentType "application/json" -Body '{"min_pages": 500}'
```

Expected Response:
```json
{
  "class": "Book",
  "method": "count_long_books",
  "status": "executed",
  "result": 3,
  "output": "Found 3 books with more than 300 pages\n"
}
```

---

#### 2.3 average_pages() - Get the average number of pages across all books
```bash
Invoke-WebRequest -Uri "http://localhost:8000/book/methods/average_pages/" -Method POST -ContentType "application/json" -Body '{}'
```

Expected Response:
```json
{
  "class": "Book",
  "method": "average_pages",
  "status": "executed",
  "result": 287.5,
  "output": "Average pages across all books: 287.50\n"
}
```

---

## LIBRARY METHODS

### 1. Instance Methods (requires library_id)

#### 1.1 info() - Get library information
```bash
Invoke-WebRequest -Uri "http://localhost:8000/library/1/methods/info/" -Method POST -ContentType "application/json" -Body '{}'
```

Expected Response:
```json
{
  "library_id": 1,
  "method": "info",
  "status": "executed",
  "result": "{'name': 'Central Library', 'address': '123 Main St'}",
  "output": "Library: Central Library\nAddress: 123 Main St\n"
}
```

---

#### 1.2 has_address() - Check if library has an address
```bash
Invoke-WebRequest -Uri "http://localhost:8000/library/1/methods/has_address/" -Method POST -ContentType "application/json" -Body '{}'
```

Expected Response:
```json
{
  "library_id": 1,
  "method": "has_address",
  "status": "executed",
  "result": "True",
  "output": "Library 'Central Library' has address: True\n"
}
```

---

### 2. Class-Level Methods (no library_id required)

#### 2.1 count_libraries() - Count total libraries
```bash
Invoke-WebRequest -Uri "http://localhost:8000/library/methods/count_libraries/" -Method POST -ContentType "application/json" -Body '{}'
```

Expected Response:
```json
{
  "class": "Library",
  "method": "count_libraries",
  "status": "executed",
  "result": 5,
  "output": "Total libraries in database: 5\n"
}
```

---

#### 2.2 find_by_name(name) - Find library by name (partial match)
```bash
# Find library with "Central" in the name
Invoke-WebRequest -Uri "http://localhost:8000/library/methods/find_by_name/" -Method POST -ContentType "application/json" -Body '{"name": "Central"}'

# Find library with "Public" in the name
Invoke-WebRequest -Uri "http://localhost:8000/library/methods/find_by_name/" -Method POST -ContentType "application/json" -Body '{"name": "Public"}'

# Find library with "Main" in the name
Invoke-WebRequest -Uri "http://localhost:8000/library/methods/find_by_name/" -Method POST -ContentType "application/json" -Body '{"name": "Main"}'
```

Expected Response (found):
```json
{
  "class": "Library",
  "method": "find_by_name",
  "status": "executed",
  "result": {
    "id": 1,
    "name": "Central Library",
    "address": "123 Main St"
  },
  "output": "Found library: Central Library\n"
}
```

Expected Response (not found):
```json
{
  "class": "Library",
  "method": "find_by_name",
  "status": "executed",
  "result": null,
  "output": "Library with name containing 'Unknown' not found\n"
}
```

---

## Testing Tips

### Create test data first

Create books and libraries before testing methods:

```bash
# Create a library
Invoke-WebRequest -Uri "http://localhost:8000/library/" -Method POST -ContentType "application/json" -Body @'
{
  "name": "City Library",
  "address": "456 Oak Avenue"
}
'@

# Create books
Invoke-WebRequest -Uri "http://localhost:8000/book/" -Method POST -ContentType "application/json" -Body @'
{
  "title": "Python Programming",
  "pages": 450,
  "release": "2020-01-15",
  "locatedIn": 1,
  "writtenBy": []
}
'@

Invoke-WebRequest -Uri "http://localhost:8000/book/" -Method POST -ContentType "application/json" -Body @'
{
  "title": "Web Development",
  "pages": 350,
  "release": "2021-06-20",
  "locatedIn": 1,
  "writtenBy": []
}
'@
```

### View API Documentation

Open your browser and navigate to:
```
http://localhost:8000/docs
```

This will show the Swagger UI with all available endpoints.

### Alternative: Use curl (if using Git Bash or WSL)

```bash
# Book hello_world
curl -X POST "http://localhost:8000/book/1/methods/hello_world/" \
  -H "Content-Type: application/json" \
  -d '{}'

# Book calculate_discount
curl -X POST "http://localhost:8000/book/1/methods/calculate_discount/" \
  -H "Content-Type: application/json" \
  -d '{"discount": 0.2}'

# Book get_longest
curl -X POST "http://localhost:8000/book/methods/get_longest/" \
  -H "Content-Type: application/json" \
  -d '{}'

# Library info
curl -X POST "http://localhost:8000/library/1/methods/info/" \
  -H "Content-Type: application/json" \
  -d '{}'

# Library count_libraries
curl -X POST "http://localhost:8000/library/methods/count_libraries/" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Summary

**Book Methods:**
- Instance: `hello_world()`, `calculate_discount(discount)`, `is_long(min_pages)`, `is_old(years)`
- Class-level: `get_longest()`, `count_long_books(min_pages)`, `average_pages()`

**Library Methods:**
- Instance: `info()`, `has_address()`
- Class-level: `count_libraries()`, `find_by_name(name)`

All methods capture print output and return results in a structured JSON format!
