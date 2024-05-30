// This is a template example to list the name of the classes
package tests.structural.library.output.java;
import java.util.List;
import java.util.ArrayList;

public class Author {
    private String email;
    private String name;

    public Author (String email, String name) {
        this.email = email;
        this.name = name;
    }

    public String getEmail() {
        return this.email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }
}