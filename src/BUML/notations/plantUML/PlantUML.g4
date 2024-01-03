grammar PlantUML;

// Parser rules
domainModel         : Start NL
                      element*
                      End
                      ;

element             : skinParam | class | relationship ;

skinParam           : 'skinparam' 'groupInheritance' INT NL ;

class               : (abstract | 'class') ID extends? '{' NL
                      (attribute | method)*
                      '}' NL ;

abstract            : 'abstract' 'class'? ;

relationship        : association | inheritance ;

association         : ID cardinality?
                      (bidirectional | unidirectional | aggregation | composition)
                      cardinality? ID (':' ID)? NL
                      ;

bidirectional       : '--' ;

unidirectional      : nav_l='<'? '--' nav_r='>'? ;

aggregation         : (aggr_l='o'? | '<'?) '--' ('>'? | aggr_r='o'?) ;

composition         : (comp_l='*'? | '<'?) '--' ('>'? | comp_r='*'?) ;

inheritance         : ID (inh_left='<|--' | '--|>') ID NL ;

extends             : 'extends' ID ;

cardinality         : '"' min=cardinalityVal ('..' max=cardinalityVal)? '"' ;

cardinalityVal      : INT | ASTK ;

attribute           : visibility? ID ':' primitiveData NL ;

method              : visibility? modifier? 'void'? ID '()' NL ;

visibility          : '#' | '-' | '~' | '+' ;

primitiveData       : 'int' | 'float' | 'str' | 'bool' | 'time' | 'date' | 'datetime' | 'timedelta' ;

modifier            : '{static}' | '{abstract}' ;

Start               : '@startuml' ;

End                 : '@enduml' ;

// Lexer rules
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
INT             : [0-9]+ ;
ASTK            : '*' ;
DOUBLE_QUOTE    : '"' 'hola' '"';
WS              : (' ' | '\t')+ -> skip ;
NL              :  ('\r'? '\n')+ ;
//STRING          : '"' .*? '"'  ;