grammar ll_arch;

architecture        : 'LLA' '{'
                      'clusters' '{' cluster+ '}'
                      'apps' '{' application+ '}'
                      'containers' '{' container+ '}'
                      '}'
                      ;

cluster             : privateCluster | publicCluster ;

privateCluster      : 'public_cluster' '{' 'name' ':' ID '}' ;

publicCluster       : 'private_cluster' '{' 'name' ':' ID '}' ;

application         : 'application' '{' 
                      'name' ':' ID ','
                      'cpu_required' ':' INT ','
                      'memory_required' ':' INT ','
                      'image' ':' STRING ','
                      'components' ':' component+
                      '}'
                      ;

component           : '[' STRING (',' STRING)* ']' ;

container           : 'container' '{'
                      'application' ':' STRING ','
                      'cluster' ':' STRING ','
                      'cpu_limit' ':' INT ','
                      'memory_limit' ':' INT ','
                      'instances' ':' INT
                      '}'
                      ;

WS                  : [ \t\r\n]+ -> skip ;

ML_COMMENT          : '/*' .*? '*/' -> skip ;

SL_COMMENT          : '//' ~[\r\n]* -> skip ;

INT                 : [0-9]+ ;

ID                  : [a-zA-Z_][a-zA-Z0-9_]* ;

STRING              : '"' .*? '"' ;