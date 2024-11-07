grammar PlantUML;

// Parser rules
domainModel         : Start NL
                      element*
                      End
                      ;

element             : (skinParam | class | relationship | enumeration) NL+ ;

skinParam           : 'skinparam' 'groupInheritance' INT NL ;

class               : (abstract | 'class') ID extends? '{' NL*
                      (attribute | method)*
                      '}' ;

abstract            : 'abstract' 'class'? ;

relationship        : association | inheritance ;

association         : ID c_left=cardinality?
                      (bidirectional | unidirectional | aggregation | composition)
                      c_right=cardinality? ID (':' ID)?
                      ;

bidirectional       : '--' ;

unidirectional      : nav_l='<'? '--' nav_r='>'? ;

aggregation         : (aggr_l='o'? | '<'?) '--' ('>'? | aggr_r='o'?) ;

composition         : (comp_l='*'? | '<'?) '--' ('>'? | comp_r='*'?) ;

inheritance         : ID (inh_left='<|--' | '--|>') ID ;

extends             : 'extends' ID ;

cardinality         : D_QUOTE min=cardinalityVal ('..' max=cardinalityVal)? D_QUOTE ;

cardinalityVal      : INT | ASTK ;

attribute           : visibility? ID ':' dType NL ;

method              : visibility? modifier? name=ID '('
                      (parameter (',' parameter)?)?
                      ')' (':' dType)? NL ;

parameter           : name=ID ':' dType ('=' value)? ;

value               : D_QUOTE? (ID | INT | FLOAT) D_QUOTE? ;

dType               : primitiveData | ID ;

enumeration         : 'enum' ID '{' NL
                      enumLiteral*
                      '}' ;

enumLiteral         : ID NL ;

visibility          : '#' | '-' | '~' | '+' ;

primitiveData       : 'int' | 'float' | 'str' | 'string' | 'bool' | 'time' | 'date' | 'datetime' | 'timedelta' ;

modifier            : '{static}' | '{abstract}' ;

Start               : '@startuml' ;

End                 : '@enduml' ;

// Lexer rules
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
INT             : [0-9]+ ;
FLOAT           : [0-9]+ '.' [0-9]+ ;
ASTK            : '*' ;
WS              : (' ' | '\t')+ -> skip ;
NL              :  ('\r'? '\n')+ ;
D_QUOTE         : '"' ;
