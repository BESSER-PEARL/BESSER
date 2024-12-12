// This is a template example to list the name of the classes
package tests.structural.library.output.java;
import java.util.List;
import java.util.ArrayList;

public class Author {
    private String name;
    private String email;

    public Author (String name, String email) {
        this.name = name;
        this.email = email;
    }



    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return this.email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}