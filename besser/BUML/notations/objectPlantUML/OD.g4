grammar OD;

Start               : '@startuml' ;

End                 : '@enduml' ;


// Entry point for parsing
objectDiagram: Start
(objectDeclaration | linkDeclaration)+
 End;



// Object declarations
objectDeclaration: 'Object' objectName (':' className)? propertiesBlock? ;

objectName                  : IDENTIFIER ;
className                   : IDENTIFIER ;
propertiesBlock             : '{' property* '}' ;

property                    : propertyName ':' propertyValue ;
propertyName                : IDENTIFIER ;
propertyValue               : (IDENTIFIER | STRING | NUMBER | DATE)+ ;


// Link declarations
linkDeclaration: linkObjectName linkType linkObjectName (':' linkName)? ;

linkObjectName: IDENTIFIER;
linkType: ('<|--'|'o--' |'--' | '..' | '-->' | '..>') ;
linkName: STRING |IDENTIFIER ;
// Lexer rules
IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]* ;
STRING: '"' ( ~["\\] | '\\' (["\\/bfnrt] | 'u' [0-9a-fA-F]{4}) )* '"' ;
NUMBER: [0-9]+ ('.' [0-9]+)? ;
WHITESPACE: [ \t\r\n]+ -> skip ;
DATE: NUMBER '/' NUMBER '/' NUMBER ;