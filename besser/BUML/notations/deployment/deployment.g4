grammar deployment;

architecture        : 'Deployment model' '{'
                      'applications' '{' application+ '}'
                      'services' '{' service+ '}'
                      'containers' '{' container+ '}'
                      'deployments' '{' deployment+ '}'
                      ('regions' '{' region+ '}')?
                      'clusters' '{' cluster+ '}'
                      '}'
                      ;

application         : '->' 
                      'name' ':' ID ','
                      'image' ':' STRING ','
                      'port' ':' INT ','
                      'cpu_required' ':' INT 'm' ','
                      'memory_required' ':' INT 'Mi' ','
                      'domain_model' ':' STRING
                      ;

service             : '->'
                      'name' ':' ID ','
                      'port' ':' INT ','
                      'target_port' ':' INT ','
                      'protocol' ':' protocol ','
                      'type' ':' service_type (','
                      'app_name' ':' app=ID)?
                      ;

container           : '->'
                      'name' ':' ID ','
                      'app_name' ':' ID ','
                      'cpu_limit' ':' INT 'm' ','
                      'memory_limit' ':' INT 'Mi'
                      ;

deployment          : '->'
                      'name' ':' ID ','
                      'replicas' ':' INT ','
                      'containers' ':' '['ID (','ID)? ']'
                      ;

region              : '->'
                      'name' ':' ID_REG (','
                      'zones' ':' '['ID_REG (',' ID_REG)? ']')?
                      ;

cluster             : privateCluster | publicCluster ;

publicCluster       : '->' 'public_cluster' 
                      'name' ':' ID ','
                      'number_of_nodes' ':' INT ','
                      'provider' ':' provider ','
                      'config_file' ':' STRING ','
                      service_list ','
                      deployment_list ','
                      region_list (','
                      'net_config' ':' boolean)? (','
                      'networks' ':' '['ID (',' ID)? ']')? (','
                      'subnetworks' ':' '['ID (',' ID)? ']')?
                      ;

privateCluster      : 'private_cluster' '{' 'name' ':' ID '}' ;

service_list       : 'services' ':' '['ID (',' ID)? ']' ;

deployment_list     : 'deployments' ':' '['ID (',' ID)? ']' ;

region_list         : 'regions' ':' '['ID_REG (',' ID_REG)? ']' ;

protocol            : 'HTTP' | 'HTTPS' | 'TCP' | 'UDP' | 'ALL' ;

service_type        : 'lb' | 'ingress' | 'egress' ;

provider            : 'google' | 'aws' | 'azure' | 'other' ;

boolean             : 'True' | 'False';

WS                  : [ \t\r\n]+ -> skip ;

ML_COMMENT          : '/*' .*? '*/' -> skip ;

SL_COMMENT          : '//' ~[\r\n]* -> skip ;

INT                 : [0-9]+ ;

ID                  : [a-zA-Z_][a-zA-Z0-9_]* ;

ID_REG              : [a-zA-Z_][a-zA-Z0-9_-]* ;

STRING              : '"' .*? '"' ;