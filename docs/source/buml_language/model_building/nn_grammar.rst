Grammar for NN
=============================

Neural networks (NNs) can be described using various notations.
we have designed a textual notation for their definition, supported by a grammar we
developed to instantiate the concepts of the metamodel.



The grammar for NN is developed using ANTLR and is shown below:

The primary rule, neuralNetwork, allows for specifying a name (ID), layers, 
sub-neural networks, tensor operations, modules to represent the sequential 
architecture of the NN, parameters, along with datasets for training and testing. 
We employ specific operators in ANTLR to enforce cardinalities in our metamodel: 
the ’+’ operator indicates one or more occurrences, ’ ?’ denotes zero or one, 
and ’*’ represents zero or more.
A key rule in the grammar is layer, where each layer must have a defined name (ID) 
and a set of parameters, which vary depending on the layer type. 
For every layer type, we define a dedicated parser rule. For instance, the linear 
rule contains general layer parameters (defined by the layerParams rule), such as 
the activation function, along with specific parameters for linear layers, such as 
in_features and out_features.
Our language supports the definition of NNs using the modules rule, which enables 
the user to specify the order of layers, sub-neural networks, and tensor operations 
for the NN.
Finally, we define the rule for specifying the training dataset, including the name 
(ID), path_data, task_type, and other parameters as defined in the metamodel. 
The complete grammar is available in the project repository.

.. code-block:: console

    grammar NN;

    // Parser rules
    neuralNetwork       : ID ':'
                        'layers' ':' layer+
                        ('sub_nn' ':' sub_nn+)*
                        ('tensor_ops' ':' tensorOp+)*
                        'modules' ':' modules
                        'params' ':' parameters
                        trainingDataset
                        testDataset
                        ;

    parameters          : 'batch_size' '=' INT
                        'epochs' '=' INT
                        'learning_rate' '=' DOUBLE
                        'optimiser' '=' STRING
                        'metrics' '=' strList
                        'loss_function' '=' lossFunction
                        'weight_decay' '=' DOUBLE
                        ;

    layer               : '-' ID ':'
                        (generalLayer | rnn | cnn | layerModifier)
                        ;

    layerParams         : ('actv_func' '=' activityFuncType)?
                        ('name_layer_input' '=' STRING)?
                        ('input_reused' '=' BOOL)?
                        ;

    generalLayer        : linear | flatten ;

    linear              : 'type' '=' 'Linear'
                        layerParams
                        'in_features' '=' INT
                        'out_features' '=' INT
                        ;

    flatten             : 'type' '=' 'Flatten'
                        layerParams
                        ('start_dim' '=' INT)?
                        ('end_dim' '=' INT)?
                        ;

    rnn                 : 'type' '=' 'RNN' ;

    cnn                 : convolutional | pooling ;

    cnnParams           : ('kernel_dim' '=' kernel=intList)?
                        ('stride_dim' '=' stride=intList)?
                        ('padding_type' '=' paddingType)?
                        ('padding_amount' '=' INT)?
                        ;

    convolutional       : 'type' '=' conv_type=('Conv1D' | 'Conv2D' | 'Conv3D')
                        layerParams
                        'in_channels' '=' INT
                        'out_channels' '=' INT
                        cnnParams
                        ;

    pooling             : 'type' '=' 'Pooling'
                        layerParams
                        'pooling_type' '=' poolingType
                        'dimension' '=' dimensionality
                        cnnParams
                        ('output_dim' '=' intList)?
                        ;

    layerModifier       : dropout | normalisation ;

    dropout             : 'type''=' 'Dropout'
                        layerParams
                        'rate' '=' DOUBLE
                        ;

    normalisation       : ;

    lossFunction        : 'crossentropy' | 'binary_crossentropy' | 'mse' ;

    sub_nn              : '-' ID ':'
                        'layers' ':'
                        layer+
                        ;

    trainingDataset     : 'TrainingDataset' ':' 
                        'name' '=' ID
                        'path_data' '=' STRING
                        'task_type' '=' taskType
                        'input_format' '=' inputFormat
                        'image' '=' intList
                        'labels' '=' '{' label ',' label (',' label)? '}'
                        ;

    testDataset         : 'TestDataset' ':'
                        'name' '=' ID
                        'path_data' '=' STRING
                        ;

    label               : '(' 'col' '=' STRING ',' 'label' '=' STRING ')' ;

    tensorOp            : '-' 'name' '=' ID
                        'type' '=' tensorOpType
                        ('concatenate_dim' '=' INT)?
                        ('layers_of_tensors' '=' intStrList)?
                        ('reshape_dim' '=' reshape=intList)?
                        ('transpose_dim' '=' transpose=intList)?
                        ('permute_dim' '=' permute=intList)?
                        ('after_activ_func' '=' after_ativ=BOOL)?
                        ('input_reused' '=' input_ref=BOOL)?
                        ;

    modules             : ('-' ID)+ ;

    intList             : '[' INT (',' INT)* ']' ;

    strList             : '[' STRING (',' STRING)* ']' ;

    intStrList          : '[' (INT|STRING) (',' (INT|STRING))* ']' ;

    activityFuncType    : 'relu'
                        | 'leaky_relu'
                        | 'sigmod'
                        | 'softmax'
                        | 'tanh'
                        | 'None'
                        ;

    tensorOpType        : 'reshape' 
                        | 'concatenate' 
                        | 'multiply'
                        | 'matmultiply'
                        | 'permute'
                        |'transpose'
                        ;

    taskType            : 'binary' | 'multi_class' | 'regression' ;

    inputFormat         : 'csv' | 'images' ;

    paddingType         : 'valid' | 'same' ;

    poolingType         : 'average' | 'max' | 'adaptive_average' | 'adaptive_max' ;

    dimensionality      : '1D' | '2D' | '3D' ;

    // Lexer rules
    ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
    INT             : [0-9]+ ;
    BOOL            : 'True' | 'False' ;
    WS              : [ \t\r\n]+ -> skip ;
    STRING          : '"' .*? '"'  ;
    DOUBLE          : [0-9]+ '.' [0-9]* ;




A textual example of the neural network (NN) model is shown below. 
The model definition begins by specifying the NN’s name (my_model). 
Next, the layers are defined outlining three layers (l1, l2, and l3), 
with l1 and l3 being 2D Convolutional layers, and l2 as a Pooling layer. Then, 
the modules definition specifies the order of the layers.
Finally, hyperparameters are defined, such as the “adam” optimiser.
The full textual model can be accessed in the project repository.

.. code-block:: console

    my_model:
    layers:
        - l1:
            type=Conv2D
            actv_func=relu
            in_channels=3
            out_channels=32
            kernel_dim=[3, 3]
        - l2:
            type=Pooling
            pooling_type=max
            dimension=2D
            kernel_dim=[2, 2]
        - l3:
            type=Conv2D
            actv_func=relu
            in_channels=32
            out_channels=64
            kernel_dim=[3, 3]
    modules:
        - l1 - l2 - l3 - l4 - l5 - l6 - l7 - l8
    config:
        batch_size=32
        epochs=10
        learning_rate=0.001
        optimiser="adam"
        metrics=["f1-score"]
        loss_function=crossentropy


