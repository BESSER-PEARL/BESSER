// This is a template example to list the name of the classes
package tests.structural.library.output.java;
import java.util.List;
import java.util.ArrayList;

public class Library {
    private String name;
    private String address;
    private List<Book> books;
  

    public Library (String name, String address) {
        this.name = name;
        this.address = address;
        this.books = new ArrayList<>();
    }

    public Library (String name, String address, ArrayList<Book> books) {
        this.name = name;
        this.address = address;
        this.books = books;
    }

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAddress() {
        return this.address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public List<Book> getBooks() {
        return this.books;
    }

    public void addBook(Book book) {
        books.add(book);
    }
}