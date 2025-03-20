grammar NN;

// Parser rules
neuralNetwork       : ID ':'
                      'layers' ':' layer+
                      ('sub_nn' ':' sub_nn+)*
                      ('tensor_ops' ':' tensorOp+)*
                      'modules' ':' modules
                      ('config' ':' parameters)?
                      trainingDataset?
                      testDataset?
                      ;

parameters          : 'batch_size' '=' INT
                      'epochs' '=' INT
                      'learning_rate' '=' DOUBLE
                      'optimiser' '=' STRING
                      'metrics' '=' strList
                      'loss_function' '=' lossFunction
                      'weight_decay' '=' DOUBLE
                      'momentum' '=' DOUBLE
                       ;

layer               : '-' ID ':'
                      (generalLayer | rnn | cnn | layerModifier)
                      ;

layerParams         : ('actv_func' '=' activityFuncType)?
                      ('name_layer_input' '=' STRING)?
                      ('input_reused' '=' BOOL)?
                      ;

generalLayer        : linear | flatten | embedding;

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

embedding           : 'type' '=' 'Embedding'
                      layerParams
                      'num_embeddings' '=' INT
                      'embedding_dim' '=' INT
                      ;

rnn                 : 'type' '=' rnn_type=('SimpleRNN' | 'LSTM' | 'GRU')
                      layerParams
                      'return_type' '=' returnTypeRRN
                      ('input_size' '=' i_size=INT)?
                      'hidden_size' '=' INT
                      ('bidirectional' '=' bid=BOOL)?
                      ('dropout' '=' dout=DOUBLE)?
                      ('batch_first' '=' b_first=BOOL)?
                      ;

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

intStrList          : '[' (INT|STRING|ID) (',' (INT|STRING|ID))* ']' ;

activityFuncType    : 'relu'
                      | 'leaky_relu'
                      | 'sigmoid'
                      | 'softmax'
                      | 'tanh'
                      | 'None'
                      ;

returnTypeRRN       : 'last' | 'full' | 'hidden' ;

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
BOOL            : 'True' | 'False' ;
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
INT             : '-'?[0-9]+ ;
WS              : [ \t\r\n]+ -> skip ;
STRING          : '"' .*? '"'  ;
DOUBLE          : [0-9]+ '.' [0-9]* ;
