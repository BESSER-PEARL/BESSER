grammar OCLs;

//oclFile: contextDeclaration;

//contextDeclaration: CONTEXT ID;



// Top-level constructs
oclFile: (contextDeclaration| expression  )* ;

// Context Declarations
contextDeclaration:
     CONTEXT ID (COLON type)? LBRACE? constraint* RBRACE? DoubleCOLON? functionCall? COLON? type?  LPAREN? ID? RPAREN? COLON? (DERIVE |BODY| Init | PRE | POST| Def)? COLON? expression? #ContextExp
 ;

constraint: (INV | PRE | POST) ID? COLON expression SEMI? ;
functionCall: ID LPAREN (SingleQuote? expression SingleQuote? COMMA?)* RPAREN | ID LPAREN (ID COLON ID)* RPAREN ;

type: BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE | OCLANY | OCLVOID | collectionType | userDefinedType|SET ;
collectionType: SET LT type GT | BAG LT type  GT| SEQUENCE LT type GT | ORDEREDSET LT type GT;
userDefinedType: ID ;

expression:
           binaryExpression expression? #binary
          | unaryExpression expression? #unary
          | IF expression THEN expression ELSE expression ENDIF  #if
          | primaryExpression  (DOT ID)* DOT OCLISTYPEOF LPAREN type RPAREN expression? #OCLISTYPEOF
          | primaryExpression  (DOT ID)* DOT OCLASTYPE LPAREN type RPAREN expression? #OCLASTYPE
          | primaryExpression  (DOT ID)* DOT OCLISKINDOF LPAREN type RPAREN expression? #OCLISKINDOF

          | primaryExpression?  (DOT ID)* Arrow ISEMPTY LPAREN RPAREN expression? RPAREN* #ISEMPTY
          | primaryExpression?  (DOT ID)* Arrow SUM LPAREN RPAREN expression? RPAREN* #SUM
          | primaryExpression?  (DOT ID)* Arrow SIZE LPAREN RPAREN expression? RPAREN* #SIZE

          |  Arrow? INCLUDES LPAREN expression RPAREN expression? RPAREN* #INCLUDES
          |  Arrow? EXCLUDES LPAREN expression RPAREN  expression? RPAREN* #EXCLUDES
          |  Arrow? LPAREN* SEQUENCE LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE* RPAREN* expression? #SEQUENCE
          |  Arrow? LPAREN* SUBSEQUENCE LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RPAREN* RBRACE*  expression? #SUBSEQUENCE
          | Arrow? ALLINSTANCES LPAREN+ expression RPAREN+ expression?  #ALLINSTANCES

          | Arrow? LPAREN* ORDEREDSET LBRACE (SingleQuote? expression SingleQuote? COMMA?)* RBRACE RPAREN* expression?  #ORDEREDSET
          | Arrow? LPAREN* SUBORDEREDSET LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE* RPAREN* expression? RPAREN  #SUBORDEREDSET

          | Arrow? LPAREN* SET LPAREN* LBRACE* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE*  RPAREN* expression?  #SET
          | Arrow? LPAREN* BAG LPAREN* LBRACE* (SingleQuote? expression SingleQuote? COMMA?)*  RBRACE* RPAREN* expression?  #BAG

          | Arrow PREPEND LPAREN+ (SingleQuote? expression SingleQuote? COMMA?)* RPAREN+ expression? #PREPEND
          | Arrow LAST LPAREN RPAREN+ expression? #LAST
          | Arrow APPEND LPAREN (SingleQuote? expression SingleQuote? COMMA?)*  RPAREN+ expression?   #APPEND
          | Arrow? (FORALL | EXISTS | SELECT | COLLECT) LPAREN (ID (COLON ID) COMMA?)+ PIPE expression RPAREN expression? #COLLECTION
          | Arrow? (FORALL | EXISTS | SELECT | COLLECT) LPAREN expression RPAREN expression? #CollectionExpressionVariable


          | Arrow SYMMETRICDIFFERENCE LPAREN expression RPAREN+ expression? #SYMMETRICDIFFERENCE
          | Arrow FIRST LPAREN RPAREN expression?  #FIRST
          | Arrow DERIVE LPAREN RPAREN expression?  #DERIVE
          | Arrow UNION LPAREN expression RPAREN  expression?#UNION
          | Def COLON expression #defExp
          | LPAREN*  primaryExpression?  (DOT ID)* operator? primaryExpression?  (DOT ID)+ expression? #PrimaryExp
          | primaryExpression  (DOT)* ID* functionCall operator? expression?  #funcCall
          | operator numberORUserDefined expression? #op
          | Arrow expression #arrowexp
          | NUMBER expression?  #number
          | Arrow? functionCall expression? #PredefinedfunctionCall
          | primaryExpression expression? #ID
          | SingleQuote expression DOT? SingleQuote DOT? expression? #SingleQuoteExp
          | DoubleDots expression #doubleDots
;



binaryExpression:  primaryExpression  (DOT ID)* operator primaryExpression (DOT ID)* ;
unaryExpression: (NOT | MINUS) primaryExpression ;

operator: EQUAL | NOTEQUAL| LT | LE | GT | GE | PLUS | MINUS | EMPTYSTRING | Divide | AND | OR | XOR | IMPLIES ; // Added 'xor' and 'implies'

numberORUserDefined: NUMBER |primaryExpression (DOT primaryExpression)* | expression |SingleQuote? expression SingleQuote? ;

primaryExpression: literal | SELF | functionCall | LPAREN expression RPAREN | ID  ;

literal: NUMBER | STRING_LITERAL | BOOLEAN_LITERAL | NULL ;
// Function and Property Calls


CONTEXT: 'context';
// Keywords
INV: 'inv' ;
PRE: 'pre' ;
POST: 'post' ;
SELF: 'self' ;
FORALL: 'forAll' ;
EXISTS: 'exists' ;
SELECT: 'select' ;
COLLECT: 'collect' ;
OCLANY: 'OclAny' ;
OCLVOID: 'OclVoid' ;
WS: [ \t\r\n]+ -> skip ;


// Symbols
DoubleDots: '..';
DoubleCOLON: '::';
LPAREN: '(' ;
RPAREN: ')' ;
LBRACE: '{' ;
RBRACE: '}' ;
SEMI: ';' ;
COLON: ':' ;
COMMA: ',' ;
DOT: '.' ;
EQUAL: '=' ;
SingleQuote: '\'';
BOOLEAN_TYPE: 'Boolean' ;
INTEGER_TYPE: 'Integer' ;
REAL_TYPE: 'Real' ;
STRING_TYPE: 'String' ;
IF: 'if' ;
THEN: 'then' ;
ELSE: 'else' ;
ENDIF: 'endif' ;
AND: 'and' ;
OR: 'or' ;
NOT: 'not' ;
NOTEQUAL: '<>' ;
LT: '<' ;
LE: '<=' ;
GT: '>' ;
GE: '>=' ;
PIPE: '|' ;
SET: 'Set' | 'set';
BAG: 'Bag';
SEQUENCE: 'Sequence';
ORDEREDSET: 'OrderedSet';
MINUS: '-';
PLUS: '+';
Divide: '/';
EMPTYSTRING: ' ';
XOR: 'xor';
IMPLIES: 'implies';
OCLASTYPE: 'oclAsType';
OCLISTYPEOF: 'oclIsTypeOf';
OCLISKINDOF: 'oclIsKindOf';
ALLINSTANCES: 'allInstances';
ISEMPTY: 'isEmpty';
SUM: 'sum';
SIZE: 'size';
INCLUDES: 'includes';
EXCLUDES: 'excludes';
SUBSEQUENCE: 'subSequence';
SUBORDEREDSET: 'subOrderedSet';
PREPEND: 'prepend';
LAST: 'last';
APPEND: 'append';
SYMMETRICDIFFERENCE: 'symmetricDifference';
FIRST: 'first';
DERIVE: 'derive';
BODY: 'body';
Init: 'init';
UNION: 'union';
NULL: 'null';
LET: 'let';
IN: 'in';
Arrow: '->' | '→';
Def: 'def';

// Basic tokens
ID: [a-zA-Z_][a-zA-Z0-9_]* ;

NUMBER: [0-9]+ ('.' [0-9]+)? ;
STRING_LITERAL: '"' ( ~["\\] | '\\' . )* '"' ;
BOOLEAN_LITERAL: 'true' | 'false';
COMMENT: '/*' .*? '*/' -> skip ;
LINE_COMMENT: '//' ~[\r\n]* -> skip ;