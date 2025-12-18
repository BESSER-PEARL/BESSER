grammar BESSERActionLanguage;

function_definition: 'def' name=ID '(' (params+=parameter (',' params+=parameter)*)? ')' '->' return_type=type body=block ;
parameter: name=ID ':' declared_type=type ('=' expr=expression)?;


/********************************************************************************************************************
 *                                               STATEMENTS                                                         *
 *******************************************************************************************************************/

statements: cond_loop | for | condition | return | function_definition | expression ';';

cond_loop: while | do_while ;
while: 'while' '(' cond=expression ')' body=block;
do_while: 'do' body=block 'while' '(' cond=expression ')';
for: 'for' '(' iterators+=iterator (',' iterators+=iterator)* ')' body=block ;
iterator: var_name=symbol 'in' sequence=expression ;

conditional_branch: block | condition;
block: '{' stmts+=statements* '}';
condition: 'if' '(' cond=expression ')' then=block ('else' elze=conditional_branch)? ;

return: 'return' expr=expression ';';

type:   single_type |
        sequence_type |
        function_type
;
single_type:
        any_type |
        classifier_type |
        real_type |
        string_type |
        int_type |
        bool_type
;
sequence_type: the_type=single_type '[]';
function_type: '[' params_type+=type (',' params_type+=type)* ']' '->' return_type=type;
any_type: 'any';
classifier_type: name=ID;
real_type: 'float' ;
string_type: 'string';
int_type: 'int';
bool_type: 'bool';




/********************************************************************************************************************
 *                                               EXPRESSION                                                         *
 *******************************************************************************************************************/

expression: assignment;

assign_target: field_access | array_access | symbol | explicit_declaration;

explicit_declaration: name=ID ':' declared_type=type;

assignment: target=assign_target '=' assignee=assignment | ternary;

ternary: expr=boolean '?' then=expression ':' elze=expression | boolean;

boolean: or;
or: left=or '||' right=and | and;
and: left=and '&&' right=equality | equality;
equality: left=comparison op=('=='|'!=') right=comparison | comparison;
comparison: left=arithmetic op=('<'|'<='|'>'|'>=') right=arithmetic | instanceof;
instanceof: instance=arithmetic 'instanceof' the_type=classifier_type | arithmetic;


arithmetic: plus_minus;
plus_minus: left=plus_minus op=('+'|'-') right=mult_div | mult_div;
mult_div: left=mult_div op=('*'|'/') right=remain | remain;
remain: left=remain '%' right=primary | primary;

primary:
    not |
    minus |
    cast |
    null_coalessing |
    selection_expression
;
not: '-' expr=primary;
minus: '!' expr=primary;
cast: '(' as=classifier_type ')' expr=expression;
null_coalessing: nullable=selection_expression '??' elze=expression;

selection_expression
    : atomic #Atom
    | receiver=selection_expression '.' field=ID #FieldAccess
    | receiver=selection_expression '[' index=expression ']' #ArrayAccess
    | receiver=selection_expression '.' name=ID '(' (args+=expression (',' args+=expression))? ')' #FunctionCall
;
field_access: receiver=selection_expression '.' field=ID;
array_access: receiver=selection_expression '[' index=expression ']';
function_call: receiver=selection_expression '.' name=ID '(' (args+=expression (',' args+=expression))? ')';

atomic:
    '(' expr=expression ')' |
    procedure_call |
    literal |
    this |
    new |
    symbol
;
procedure_call: name=ID '(' (args+=expression (',' args+=expression))? ')';
this: 'this';
new: 'new' clazz=classifier_type '(' (args+=expression (',' args+=expression))? ')';

literal:
    single_literal |
    sequence_literal |
    range_literal
;
single_literal:
    int_literal |
    string_literal |
    bool_literal |
    real_literal |
    null_literal |
    enum_literal
;
int_literal: value=INT;
string_literal: value=STR;
bool_literal: value=('true' | 'false');
real_literal: value=FLOAT;
null_literal: value='null';
enum_literal: enum=classifier_type '::' name=ID;
sequence_literal: the_type=sequence_type '{' (values+=expression (',' values+=expression)*)? '}';
range_literal: the_type=int_type '[]' '{' first=expression '..' last=expression '}';

symbol: name=ID;


// Lexer rules
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
INT             : [0-9]+ ;
FLOAT           : [0-9]+ '.' [0-9]+ ;
WS              : (' ' | '\t' | NL)+ -> skip ;
NL              :  ('\r'? '\n')+ ;
STR :
    ('"' ( ~[\\"\n\r] | '\\' [\\"] )* '"') |
    ('\'' ( ~[\\"\n\r] | '\\' [\\"] )* '\'' )
;
