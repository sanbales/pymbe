# Developer Jupyter Notebooks

The collection of Jupyter notebooks housed here are meant to walk through intended uses of PyMBE with live examples that use the API. They should explain the functionality of the codebase (even if not yet implemented) and explain why certain patterns and approaches exist. They can also use reflective methods in Python in order to show how the code is implemented and point to important unit tests.

## Functionality of pyMBE

PyMBE is not meant to compete with large system modeling IDEs or CAD tools. What it is meant for is to provide a Pythonic connection to system models that allow for them to be generated from and support workflows that involve multiple engineering tools as well as document generation.

### Browsing models

SysML v2 library models and others can become fairly complicated to newcomers. Since Python has access to libraries for visualization and graph data structures, PyMBE provides the opportunity to inspect models and show how semantic information is encoded within them. 

### Rapidly creating models from recipes

With the Python language, a series of functions can be created that build model elements at high or low levels. The intent is to start with low level patterns such as creating a single Part Definition or Interface Usage and work up to building higher level patterns such as adding a serial communications interface to a model. This helps get past a lot of boilerplate needed to make SysML v2 models functional and can help make models more productive.

### Querying and navigating models

Pythonic approaches to data handling make SysML v2 objects look like normal Python classes. For example, model_element.name can be used to access the name directly. There are also additional properties to help with navigating across v2 relationships in either for the through or reverse directions. SysML v2 attributes can be used to access the relationships with accesssors like model_element.ownedRelationship.

### Associating models with data

A major antipattern in system modeling practice was to blend models (classes meant to constrain and describe broad sets of individual objects) and data (the objects themselves) into a single model. The gathering of data from engineering teams into CAD tools added overhead and friction to the use of this data and led to resistance in adoption. With the wide array of data processing libraries available in Python, it's possible to digest a wide variety of inputs and create appropriate models. This includes using the data to find values for the usages within different definitions in SysML.

### Generating legal data from models

Models should also be able to generate data that represent legal instances of the model. For example, Part Definitions with subparts of indeterminate multiplicity should be refined to specialized instancs where the part multiplicities are fixed. Within the fixed multiplicity, there should be unique parts of the same number. Actions and State Machines should be simulated to create traces, even if this means a call to other Python frameworks to do the simulation. 

## Infrastructure in PyMBE

The functionality above relies on some infrastructure that is necessary to create the necessary Pythonic classes and their properties. That infrastructure is also described here.

### Loading models from SysML v2 REST API

The standard REST API represents model elements as JSON objects. The references between models are coded as unique ID references while primitive type data in integers, strings, booleans, and reals are attributes on the object. The PyMBE infrastructure creates and populates Python objects from the JSON data and resolves the references so that model navigation is natural and Pythonic.

### Working with derived attributes and inheritance

SysML v2, and its basis KerML, has a rich vocabulary of inheritance and overriding mechanisms. Working with the consequences of this in a natural way is essential to being able to work with models.

### Model labeling

In order to display elements in an intelligible way, models must generate sensible names and labels. There are some helper functions in PyMBE for that.

### Loading up metamodel data

The Ecore-based metamodels are automatically inspected and used to generate Python objects that align with the SysML v2 metamodel.