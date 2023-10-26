Build a model using B-UML
=========================

The following guide shows how to define the classic Library model (see following image) using the B-UML language.

.. image:: ../img/library_uml_model.jpg
  :width: 600
  :alt: Library model

To define your model using B-UML, you must first import the B-UML classes you are going to use. The following classes 
must be imported for this Library modeling example.

.. code-block:: python

    from BUML.metamodel.structural.structural import DomainModel, Class, Property, \
        PrimitiveDataType, Multiplicity, BinaryAssociation

Now, we can define the attributes, classes, relationships, and other elements of the model. For classes and attributes 
use ``Property`` and ``Class`` classes respectively. The following is the definition of the *Book* class including its attributes.

.. code-block:: python

    # Book attributes definition
    title: Property = Property(name="title", owner=None, property_type=PrimitiveDataType("str"))
    pages: Property = Property(name="pages", owner=None, property_type=PrimitiveDataType("int"))
    release: Property = Property(name="release", owner=None, property_type=PrimitiveDataType("date"))

    # Book class definition
    book: Class = Class (name="Book", attributes={title, pages, release})

Different types of relationships can be specified with B-UML such as associations (including binary associations), generalizations, 
generalization sets, and class associations. Using the ``BynaryAssociation`` class, we can specify the relationship between *Library* 
and *Book* as follows. 

.. code-block:: python

    # Library-Book association definition
    located_in_end: Property = Property(name="locatedIn", owner=None, property_type=library, multiplicity=Multiplicity(1, 1))
    has_end: Property = Property(name="has", owner=None, property_type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_association: BinaryAssociation = BinaryAssociation(name="lib-book-assoc", ends={located_in_end, has_end})

Finally, create the domain model and add the classes, relationships and other elements of the model.

.. code-block:: python

    # Domain model definition
    library_model : DomainModel = DomainModel(name="Library model", types={library, book, author}, 
                                              associations={lib_book_association, book_author_association})






.. note::
    
    For a detailed description of metamodel elements such as classes, attributes, generalizations and others, 
    please refer to the API documentation or consult other examples.