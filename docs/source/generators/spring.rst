Spring Boot Generator
=====================

BESSER provides a code generator for `Spring Boot <https://spring.io/projects/spring-boot>`_
backends. From a structural model it produces a complete, runnable Maven project:
JPA entities, Spring Data repositories, a service layer, REST controllers, and a
Maven wrapper, so the generated project builds and starts without any further setup.

**B-UML Model required**

- :doc:`../buml_language/model_types/structural`: the classes, attributes,
  associations and enumerations that become the persistence and API layers.

**Getting started**

You can either use the :doc:`../web_editor` to draw the structural model and generate
the code directly, or use the BESSER Python API as shown below. This example uses the
:doc:`../examples/library_example` as input.

.. code-block:: python

    from besser.generators.spring import SpringBackendGenerator

    generator: SpringBackendGenerator = SpringBackendGenerator(
        model=library_model,
        output_dir="output/library",
        app_name="LibraryApplication",
        package_name="com.example.library",
    )
    generator.generate()

**Configuration Parameters**

Only ``model`` is required. Every other parameter is keyword-only and has a default,
so the shortest useful call is ``SpringBackendGenerator(model, output_dir="...")``.

- ``model``: the structural model the backend is generated from.
- ``output_dir``: the directory the project is written to. When omitted, the
  generator writes to ``<cwd>/output``.
- ``spring_boot_version``: the ``spring-boot-starter-parent`` version written to
  the POM. Defaults to ``"3.4.4"``.
- ``java_version``: the ``java.version`` property of the generated POM. Defaults
  to ``"21"``.
- ``app_name``: the application name, which is also the name of the generated main
  class and of its test class. Defaults to ``"Application"``.
- ``package_name``: the root Java package of the generated sources. It is validated
  as a legal Java package name and determines the directory nesting under
  ``src/main/java``. Defaults to ``"com.example"``.
- ``group_id``: the Maven ``groupId`` of the generated project. Defaults to
  ``"com.example"``.
- ``description``: the Maven ``description`` of the generated project. Defaults to
  an empty string.

What is Generated
-----------------

- **Entities**: each class becomes a JPA ``@Entity`` with an ``@Table``, getters and
  setters, and an all-args constructor. An attribute marked as the model's identifier
  becomes ``@Id``; it additionally gets ``@GeneratedValue(strategy = GenerationType.IDENTITY)``
  only when its type maps to a Java ``Integer`` or ``Long``, since Hibernate cannot
  auto-generate a ``String`` identifier. An abstract class becomes a ``@MappedSuperclass``
  instead, and generalizations are emitted as Java ``extends``.
- **Enumerations**: each enumeration becomes a Java ``enum``, and attributes typed by
  it are annotated ``@Enumerated(EnumType.STRING)``.
- **Associations**: the multiplicity of each end decides the JPA annotation —
  ``@OneToOne``, ``@OneToMany``, ``@ManyToOne`` or ``@ManyToMany`` — with
  ``@JoinColumn`` on the owning side, ``@JoinTable`` for many-to-many, and
  ``mappedBy`` on the inverse side. A non-navigable end produces no field.
- **Repositories**: one ``I<Class>Repository extends JpaRepository`` per concrete
  class, with a derived finder for each simple attribute, plus a ``...Between``
  finder for date and time attributes.
- **Services**: an ``I<Class>Service`` interface and a ``<Class>Service``
  implementation per concrete class, providing CRUD over the repository.
- **Controllers**: a ``@RestController`` per concrete class, mapped at
  ``/api/<class>``, exposing ``GET``, ``GET /{id}``, ``POST``, ``PUT /{id}`` and
  ``DELETE /{id}``.
- **HTTP request files**: a ``.http`` file per concrete class under
  ``src/test/resources/http``, ready to run against the started application from an
  IDE or REST client.

Abstract classes take part in the entity hierarchy but get no repository, service,
controller or ``.http`` file of their own.

Output
------

For a model with an ``Author`` class, a ``Book`` class and a ``Genre`` enumeration,
generated with ``app_name="LibraryApplication"`` and
``package_name="com.example.library"``:

.. code-block:: text

    output/library/
    ├── pom.xml
    ├── mvnw
    ├── mvnw.cmd
    ├── .mvn/
    │   └── wrapper/
    │       └── maven-wrapper.properties
    └── src/
        ├── main/
        │   ├── java/com/example/library/
        │   │   ├── LibraryApplication.java
        │   │   ├── entity/
        │   │   │   ├── Author.java
        │   │   │   ├── Book.java
        │   │   │   └── Genre.java
        │   │   ├── repository/
        │   │   │   ├── IAuthorRepository.java
        │   │   │   └── IBookRepository.java
        │   │   ├── service/
        │   │   │   ├── interfaces/
        │   │   │   │   ├── IAuthorService.java
        │   │   │   │   └── IBookService.java
        │   │   │   └── impl/
        │   │   │       ├── AuthorService.java
        │   │   │       └── BookService.java
        │   │   └── controller/
        │   │       ├── AuthorController.java
        │   │       └── BookController.java
        │   └── resources/
        │       └── application.properties
        └── test/
            ├── java/com/example/library/
            │   └── LibraryApplicationTests.java
            └── resources/http/
                ├── author.http
                └── book.http

How to Run the Application
--------------------------

**Requirement**: a JDK matching the ``java_version`` the project was generated with
(21 by default). Maven itself does not need to be installed — the generated project
carries a Maven wrapper.

Go to the project folder and run:

.. code-block:: bash

    # Start the application (use mvnw.cmd on Windows)
    ./mvnw spring-boot:run

The REST API is then served at `http://localhost:8080 <http://localhost:8080>`_, with
each class exposed under ``/api/<class>`` — for example
`http://localhost:8080/api/book <http://localhost:8080/api/book>`_.

The generated project is configured for an in-memory
`H2 database <https://www.h2database.com/>`_ with ``spring.jpa.hibernate.ddl-auto=update``,
so the schema is created at startup and no database has to be provisioned to try the
application out. The data is discarded when the application stops. The H2 console is
enabled at `http://localhost:8080/h2-console <http://localhost:8080/h2-console>`_.
Point ``spring.datasource.*`` in ``src/main/resources/application.properties`` at a real
database to persist beyond a run.

.. note::

   The :doc:`../web_editor` offers this generator in the **Web** group of the
   **Generate** menu, and returns the project as a ZIP archive.
