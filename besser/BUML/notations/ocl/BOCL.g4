grammar BOCL;

// =============================================================
// Top-level rules
// =============================================================
oclFile
    : contextDeclaration EOF
    | initConstraints EOF
    | preCondition EOF
    | postCondition EOF
    ;

preCondition
    : CONTEXT ID DOUBLECOLON ID LPAREN paramList? RPAREN PRE COLON expression
    ;

postCondition
    : CONTEXT ID DOUBLECOLON ID LPAREN paramList? RPAREN POST COLON expression
    ;

initConstraints
    : CONTEXT ID DOUBLECOLON ID COLON typeRef INIT COLON expression
    ;

contextDeclaration
    : CONTEXT ID constraint+
    ;

constraint
    : INV ID? COLON expression SEMI?
    ;

paramList
    : param (COMMA param)*
    ;

param
    : ID? COLON typeRef
    ;

// =============================================================
// Type system
// =============================================================
typeRef
    : primitiveType
    | collectionType
    | ID
    ;

primitiveType
    : BOOLEAN_TYPE | INTEGER_TYPE | REAL_TYPE | STRING_TYPE
    ;

collectionType
    : (SET | BAG | SEQUENCE | ORDEREDSET) LT typeRef GT
    ;

// =============================================================
// Helper rules
// =============================================================
iteratorVarDecl
    : ID (COLON ID)? (COMMA ID (COLON ID)?)*
    ;

iteratorOp
    : FORALL | EXISTS | SELECT | REJECT | COLLECT
    ;

compOp
    : EQUAL | NOTEQUAL | LT | LE | GT | GE
    ;

argList
    : expression (COMMA expression)*
    ;

// =============================================================
// Expression rule with ANTLR4 precedence climbing
// Alternatives listed FIRST have HIGHEST precedence.
// =============================================================
expression
    // --- Postfix: dot navigation and method calls (highest precedence) ---
    : expression DOT SIZE LPAREN RPAREN                                             #dotSize
    | expression DOT OCLISTYPEOF LPAREN typeRef RPAREN                              #dotOclIsTypeOf
    | expression DOT OCLASTYPE LPAREN typeRef RPAREN                                #dotOclAsType
    | expression DOT OCLISKINDOF LPAREN typeRef RPAREN                              #dotOclIsKindOf
    | expression DOT ID LPAREN argList? RPAREN                                      #dotMethodCall
    | expression DOT ID                                                             #dotNavigation
    // Fallback: `size` is also a valid attribute name when no parens follow.
    // ANTLR's longest-match keeps `dotSize` (`.size()`) winning when parens are
    // present, so `collection->size()` and `string.size()` still parse.
    // See BESSER-PEARL/BESSER#198.
    | expression DOT SIZE                                                           #dotSizeNavigation

    // --- Postfix: arrow operations ---
    | expression ARROW iteratorOp LPAREN iteratorVarDecl PIPE expression RPAREN     #arrowIterator
    | expression ARROW iteratorOp LPAREN expression RPAREN                          #arrowIteratorShort
    | expression ARROW SIZE LPAREN RPAREN                                           #arrowSize
    | expression ARROW ISEMPTY LPAREN RPAREN                                        #arrowIsEmpty
    | expression ARROW SUM LPAREN RPAREN                                            #arrowSum
    | expression ARROW INCLUDES LPAREN expression RPAREN                            #arrowIncludes
    | expression ARROW EXCLUDES LPAREN expression RPAREN                            #arrowExcludes
    | expression ARROW UNION LPAREN expression RPAREN                               #arrowUnion
    | expression ARROW FIRST LPAREN RPAREN                                          #arrowFirst
    | expression ARROW LAST LPAREN RPAREN                                           #arrowLast
    | expression ARROW PREPEND LPAREN expression RPAREN                             #arrowPrepend
    | expression ARROW APPEND LPAREN expression RPAREN                              #arrowAppend
    | expression ARROW SYMMETRICDIFFERENCE LPAREN expression RPAREN                 #arrowSymDiff
    | expression ARROW SUBSEQUENCE LPAREN expression COMMA expression RPAREN        #arrowSubSequence
    | expression ARROW SUBORDEREDSET LPAREN expression COMMA expression RPAREN      #arrowSubOrderedSet

    // --- Unary ---
    | (NOT | MINUS) expression                                                      #unaryExpr

    // --- Binary: multiplicative ---
    | expression (STAR | DIVIDE) expression                                         #mulDivExpr

    // --- Binary: additive ---
    | expression (PLUS | MINUS) expression                                          #addSubExpr

    // --- Binary: comparison ---
    | expression compOp expression                                                  #comparisonExpr

    // --- Binary: logical (lowest precedence) ---
    | expression AND expression                                                     #andExpr
    | expression XOR expression                                                     #xorExpr
    | expression OR expression                                                      #orExpr
    | expression IMPLIES expression                                                 #impliesExpr

    // --- Non-recursive: if-then-else and primary ---
    | IF expression THEN expression ELSE expression ENDIF                           #ifThenElseExpr
    | primaryExpression                                                             #primaryExpr
    ;

// =============================================================
// Primary expressions (atoms)
// =============================================================
primaryExpression
    : SELF                                                      #selfExpr
    | STRING_LITERAL                                            #stringLiteral
    | NUMBER                                                    #numberLiteral
    | BOOLEAN_LITERAL                                           #booleanLiteral
    | NULL                                                      #nullLiteral
    | ID DOUBLECOLON ALLINSTANCES LPAREN RPAREN                 #allInstancesExpr
    | DATE DOUBLECOLON ID LPAREN RPAREN                         #dateFuncExpr
    | ID LPAREN argList? RPAREN                                 #functionCallExpr
    | LPAREN expression RPAREN                                  #parenExpr
    | ID                                                        #idExpr
    ;

// =============================================================
// Lexer rules
// =============================================================

// Keywords
CONTEXT    : 'context' ;
INIT       : 'init' ;
INV        : 'inv' ;
PRE        : 'pre' ;
POST       : 'post' ;
SELF       : 'self' ;
IF         : 'if' ;
THEN       : 'then' ;
ELSE       : 'else' ;
ENDIF      : 'endif' ;
LET        : 'let' ;
IN         : 'in' ;
DEF        : 'def' ;

// Boolean operators
AND        : 'and' ;
OR         : 'or' ;
NOT        : 'not' ;
XOR        : 'xor' ;
IMPLIES    : 'implies' ;

// Collection iterators
FORALL     : 'forAll' ;
EXISTS     : 'exists' ;
SELECT     : 'select' ;
REJECT     : 'reject' ;
COLLECT    : 'collect' ;

// Collection/type operations
SIZE               : 'size' ;
ISEMPTY            : 'isEmpty' ;
SUM                : 'sum' ;
INCLUDES           : 'includes' ;
EXCLUDES           : 'excludes' ;
UNION              : 'union' ;
FIRST              : 'first' ;
LAST               : 'last' ;
PREPEND            : 'prepend' ;
APPEND             : 'append' ;
ALLINSTANCES       : 'allInstances' ;
SUBSEQUENCE        : 'subSequence' ;
SUBORDEREDSET      : 'subOrderedSet' ;
SYMMETRICDIFFERENCE: 'symmetricDifference' ;
OCLISTYPEOF        : 'oclIsTypeOf' ;
OCLASTYPE          : 'oclAsType' ;
OCLISKINDOF        : 'oclIsKindOf' ;

// Types
BOOLEAN_TYPE : 'Boolean' ;
INTEGER_TYPE : 'Integer' ;
REAL_TYPE    : 'Real' ;
STRING_TYPE  : 'String' | 'str' | 'Str' | 'string' ;
SET          : 'Set' | 'set' ;
BAG          : 'Bag' ;
SEQUENCE     : 'Sequence' ;
ORDEREDSET   : 'OrderedSet' ;
DATE         : 'Date' | 'date' ;
NULL         : 'null' ;

// Symbols (multi-char before single-char to avoid partial matches)
DOUBLECOLON : '::' ;
ARROW       : '->' | '\u2192' ;
NOTEQUAL    : '<>' ;
LE          : '<=' ;
GE          : '>=' ;
DOT         : '.' ;
LPAREN      : '(' ;
RPAREN      : ')' ;
LBRACE      : '{' ;
RBRACE      : '}' ;
SEMI        : ';' ;
COLON       : ':' ;
COMMA       : ',' ;
PIPE        : '|' ;
LT          : '<' ;
GT          : '>' ;
EQUAL       : '=' ;
PLUS        : '+' ;
MINUS       : '-' ;
STAR        : '*' ;
DIVIDE      : '/' ;

// Literals (before ID so they match first)
BOOLEAN_LITERAL : 'true' | 'false' ;
NUMBER          : [0-9]+ ('.' [0-9]+)? ;
STRING_LITERAL  : '\'' (~['\\\r\n] | '\\' .)* '\'' ;

// Identifiers (last so keywords take priority)
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;

// Skip whitespace and comments
WS           : [ \t\r\n]+ -> skip ;
COMMENT      : '/*' .*? '*/' -> skip ;
LINE_COMMENT : '//' ~[\r\n]* -> skip ;
