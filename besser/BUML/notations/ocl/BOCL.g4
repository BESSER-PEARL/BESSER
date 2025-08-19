grammar BOCL;

//oclFile: contextDeclaration;

//contextDeclaration: CONTEXT ID;



// Top-level constructs
oclFile: initConstraints| preCondition|postCondition|contextDeclaration (expression  )* ;
//context Library:: findBook(title:str) pre: self.has->size()>0
preCondition:CONTEXT ID DoubleCOLON ID LPAREN? (ID? COLON (BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE |  collectionType|SET)?)* RPAREN? PRE COLON expression;
postCondition:CONTEXT ID DoubleCOLON ID LPAREN? (ID? COLON (BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE |  collectionType|SET)?)* RPAREN? POST COLON expression;

initConstraints: CONTEXT ID DoubleCOLON ID COLON (BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE |  collectionType|SET) INIT COLON expression;

// Context Declarations
contextDeclaration:
     CONTEXT ID (COLON type)? LBRACE? constraint* RBRACE? DoubleCOLON? functionCall? COLON? type?  LPAREN? ID? RPAREN? COLON? (DERIVE |BODY| INIT | PRE | POST| Def)? COLON? expression? #ContextExp
 ;

constraint: (INV | PRE | POST) ID? COLON expression SEMI? ;
functionCall: ID LPAREN (SingleQuote? expression SingleQuote? COMMA?)* RPAREN | ID LPAREN (ID COLON ID)* RPAREN
 | LPAREN(NUMBER COMMA?)* RPAREN;

type: BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE | OCLANY | OCLVOID | collectionType | userDefinedType|SET ;
collectionType: SET LT type GT | BAG LT type  GT| SEQUENCE LT type GT | ORDEREDSET LT type GT;
userDefinedType: ID ;

expression:
          (AND | OR )? binaryExpression expression? #binary
          | unaryExpression expression? #unary
          | IF expression  #ifExp
          | THEN expression #thenExp
          | ELSE expression #elseExp
          | ENDIF  expression? #endIfExp
          | primaryExpression  (DOT ID)* DOT OCLISTYPEOF LPAREN type RPAREN expression? #OCLISTYPEOF
          | primaryExpression  (DOT ID)* DOT OCLASTYPE LPAREN type RPAREN expression? #OCLASTYPE
          | primaryExpression  (DOT ID)* DOT OCLISKINDOF LPAREN type RPAREN expression? #OCLISKINDOF

          | primaryExpression?  (DOT ID)* Arrow ISEMPTY LPAREN RPAREN expression? RPAREN* #ISEMPTY
          | primaryExpression?  (DOT ID)* Arrow SUM LPAREN RPAREN  binaryFunctionCall? expression? RPAREN* #SUM
          | primaryExpression?  (DOT ID)* (DOT | Arrow) SIZE   LPAREN RPAREN binaryFunctionCall? expression? RPAREN* #SIZE

          |  Arrow? INCLUDES LPAREN expression RPAREN expression? RPAREN* #INCLUDES
          |  Arrow? EXCLUDES LPAREN expression RPAREN  expression? RPAREN* #EXCLUDES
          |  Arrow? LPAREN* SEQUENCE LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE* RPAREN* expression? #SEQUENCE
          |  Arrow? LPAREN* SUBSEQUENCE LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RPAREN* RBRACE*  expression? #SUBSEQUENCE
          |  Arrow? ALLINSTANCES LPAREN+ expression? RPAREN+ expression?  #ALLINSTANCES

          | Arrow? LPAREN* ORDEREDSET LBRACE (SingleQuote? expression SingleQuote? COMMA?)* RBRACE RPAREN* expression?  #ORDEREDSET
          | Arrow? LPAREN* SUBORDEREDSET LBRACE* LPAREN* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE* RPAREN* expression? RPAREN  #SUBORDEREDSET

          | Arrow? LPAREN* SET LPAREN* LBRACE* (SingleQuote? expression SingleQuote? COMMA?)* RBRACE*  RPAREN* expression?  #SET
          | Arrow? LPAREN* BAG LPAREN* LBRACE* (SingleQuote? expression SingleQuote? COMMA?)*  RBRACE* RPAREN* expression?  #BAG

          | Arrow PREPEND LPAREN+ (SingleQuote? expression SingleQuote? COMMA?)* RPAREN+ expression? #PREPEND
          | Arrow LAST LPAREN RPAREN+ expression? #LAST
          | Arrow APPEND LPAREN (SingleQuote? expression SingleQuote? COMMA?)*  RPAREN+ expression?   #APPEND

          | Arrow? (FORALL | EXISTS | SELECT|REJECT | COLLECT) LPAREN (ID (COLON ID)? COMMA?)+ PIPE expression RPAREN endExpression? #COLLECTION

          | Arrow? (FORALL | EXISTS | SELECT|REJECT | COLLECT) LPAREN expression RPAREN endExpression? #CollectionExpressionVariable
//
//
          | Arrow SYMMETRICDIFFERENCE LPAREN expression RPAREN+ expression? #SYMMETRICDIFFERENCE
          | Arrow FIRST LPAREN RPAREN expression?  #FIRST
          | Arrow DERIVE LPAREN RPAREN expression?  #DERIVE
          | Arrow UNION LPAREN expression RPAREN  expression?#UNION
          | Def COLON expression #defExp
          | ID COLON ID EQUAL expression #defIDAssignmentexpression
          | LPAREN*  primaryExpression?  (DOT ID)* operator? primaryExpression?  (DOT ID)+ expression? #PrimaryExp
          | primaryExpression  (DOT)* ID* functionCall operator? expression?  #funcCall
//          | operator expression #operatorExp
          | Arrow expression #arrowexp
          | NUMBER expression?  #number
          | Arrow? functionCall expression? #PredefinedfunctionCall

          | SingleQuote expression DOT? SingleQuote DOT? expression? #SingleQuoteExp
          | DoubleDots expression #doubleDots
          | AND? OR? ID? DoubleCOLON expression #doubleCOLONs
          | operator numberORUserDefined?  #op

          | primaryExpression expression? #ID


;
endExpression:  (AND | OR)? expression;
binaryFunctionCall: operator ((primaryExpression (DOT ID)*) | NUMBER)  ;

binaryExpression:  ((primaryExpression (DOT ID)*) | NUMBER| dateLiteral)   (DOT ID)* operator (primaryExpression DoubleCOLON ID|(primaryExpression (DOT ID)*) | NUMBER| dateLiteral) ;
unaryExpression: (NOT | MINUS|PLUS|Divide|'*') expression ;
//
operator: EQUAL | NOTEQUAL| LT | LE | GT | GE | PLUS|'*' | MINUS | EMPTYSTRING | Divide | AND | OR | XOR | IMPLIES ; // Added 'xor' and 'implies'
//
numberORUserDefined: NUMBER |SingleQuote? ID LPAREN? RPAREN? SingleQuote?  ;

primaryExpression: literal | SELF | functionCall | LPAREN expression RPAREN | ID  ;

literal: NUMBER | STRING_LITERAL | BOOLEAN_LITERAL | NULL ;
dateLiteral : DATE DoubleCOLON? ('now'|'today')? LPAREN? RPAREN? DOT? 'addDays'? LPAREN? NUMBER? RPAREN?;
// Function and Property Calls


CONTEXT: 'context';
// Keywords
INIT: 'init';
INV: 'inv' ;
PRE: 'pre' ;
POST: 'post' ;
SELF: 'self' ;
FORALL: 'forAll' ;
EXISTS: 'exists' ;
SELECT: 'select' ;
REJECT: 'reject' ;
COLLECT: 'collect' ;
OCLANY: 'OclAny' ;
OCLVOID: 'OclVoid' ;
DATE: 'date' | 'Date';
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
STRING_TYPE: 'String' |'str'|'Str'|'string';
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
//Init: 'init';
UNION: 'union';
NULL: 'null';
LET: 'let';
IN: 'in';
Arrow: '->' | '→';
Def: 'def';

// Basic tokens
ID: [a-zA-Z_][a-zA-Z0-9@_]* '@pre'?;

NUMBER: [0-9]+ ('.' [0-9]+)? ;
STRING_LITERAL: '"' ( ~["\\] | '\\' . )* ID? '"'
| SingleQuote ID SingleQuote;
BOOLEAN_LITERAL: 'true' | 'false';
COMMENT: '/*' .*? '*/' -> skip ;
LINE_COMMENT: '//' ~[\r\n]* -> skip ;