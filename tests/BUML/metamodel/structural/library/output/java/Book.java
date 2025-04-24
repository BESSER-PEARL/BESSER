// This is a template example to list the name of the classes
package tests.structural.library.output.java;
import java.time.LocalDate;
import java.util.List;
import java.util.ArrayList;

public class Book {
    private int pages;
    private LocalDate release;
    private String title;
    private List<Author> authors;
              

    public Book (int pages, LocalDate release, String title) {
        this.pages = pages;
        this.release = release;
        this.title = title;
        this.authors = new ArrayList<>();
    }

    public Book (int pages, LocalDate release, String title, ArrayList<Author> authors) {
        this.pages = pages;
        this.release = release;
        this.title = title;
        this.authors = authors;
    }

    public int getPages() {
        return this.pages;
    }

    public void setPages(int pages) {
        this.pages = pages;
    }

    public LocalDate getRelease() {
        return this.release;
    }

    public void setRelease(LocalDate release) {
        this.release = release;
    }

    public String getTitle() {
        return this.title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public List<Author> getAuthors() {
        return this.authors;
    }

    public void addAuthor(Author author) {
        authors.add(author);
    }
}