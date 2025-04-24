# Generated from ./NN.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,114,459,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,
        7,6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,
        13,2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,
        20,7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,
        26,2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,
        33,7,33,1,0,1,0,1,0,1,0,1,0,4,0,74,8,0,11,0,12,0,75,1,0,1,0,1,0,
        4,0,81,8,0,11,0,12,0,82,5,0,85,8,0,10,0,12,0,88,9,0,1,0,1,0,1,0,
        4,0,93,8,0,11,0,12,0,94,5,0,97,8,0,10,0,12,0,100,9,0,1,0,1,0,1,0,
        1,0,1,0,1,0,3,0,108,8,0,1,0,3,0,111,8,0,1,0,3,0,114,8,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,148,
        8,2,1,3,1,3,1,3,3,3,153,8,3,1,3,1,3,1,3,3,3,158,8,3,1,3,1,3,1,3,
        3,3,163,8,3,1,4,1,4,1,4,3,4,168,8,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,
        1,5,1,5,1,5,1,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,3,6,188,8,6,1,6,1,6,
        1,6,3,6,193,8,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,8,
        1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,3,8,216,8,8,1,8,1,8,1,8,1,8,
        1,8,1,8,3,8,224,8,8,1,8,1,8,1,8,3,8,229,8,8,1,8,1,8,1,8,3,8,234,
        8,8,1,9,1,9,3,9,238,8,9,1,10,1,10,1,10,3,10,243,8,10,1,10,1,10,1,
        10,3,10,248,8,10,1,10,1,10,1,10,3,10,253,8,10,1,10,1,10,1,10,3,10,
        258,8,10,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
        1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,3,12,286,8,12,1,13,1,13,3,13,290,8,13,1,14,1,14,1,14,1,
        14,1,14,1,14,1,14,1,14,1,15,1,15,1,16,1,16,1,17,1,17,1,17,1,17,1,
        17,1,17,4,17,310,8,17,11,17,12,17,311,1,18,1,18,1,18,1,18,1,18,1,
        18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,
        18,1,18,1,18,1,18,1,18,1,18,1,18,3,18,339,8,18,1,18,1,18,1,19,1,
        19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,20,1,20,1,20,1,20,1,20,1,
        20,1,20,1,20,1,20,1,20,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,
        21,1,21,3,21,372,8,21,1,21,1,21,1,21,3,21,377,8,21,1,21,1,21,1,21,
        3,21,382,8,21,1,21,1,21,1,21,3,21,387,8,21,1,21,1,21,1,21,3,21,392,
        8,21,1,21,1,21,1,21,3,21,397,8,21,1,21,1,21,1,21,3,21,402,8,21,1,
        22,1,22,4,22,406,8,22,11,22,12,22,407,1,23,1,23,1,23,1,23,5,23,414,
        8,23,10,23,12,23,417,9,23,1,23,1,23,1,24,1,24,1,24,1,24,5,24,425,
        8,24,10,24,12,24,428,9,24,1,24,1,24,1,25,1,25,1,25,1,25,5,25,436,
        8,25,10,25,12,25,439,9,25,1,25,1,25,1,26,1,26,1,27,1,27,1,28,1,28,
        1,29,1,29,1,30,1,30,1,31,1,31,1,32,1,32,1,33,1,33,1,33,0,0,34,0,
        2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,
        48,50,52,54,56,58,60,62,64,66,0,12,1,0,30,32,1,0,43,45,1,0,54,56,
        2,0,110,111,113,113,1,0,80,85,1,0,86,88,1,0,89,94,1,0,95,97,1,0,
        98,99,1,0,100,101,1,0,102,105,1,0,106,108,466,0,68,1,0,0,0,2,115,
        1,0,0,0,4,140,1,0,0,0,6,152,1,0,0,0,8,167,1,0,0,0,10,169,1,0,0,0,
        12,180,1,0,0,0,14,194,1,0,0,0,16,205,1,0,0,0,18,237,1,0,0,0,20,242,
        1,0,0,0,22,259,1,0,0,0,24,271,1,0,0,0,26,289,1,0,0,0,28,291,1,0,
        0,0,30,299,1,0,0,0,32,301,1,0,0,0,34,303,1,0,0,0,36,313,1,0,0,0,
        38,342,1,0,0,0,40,351,1,0,0,0,42,361,1,0,0,0,44,405,1,0,0,0,46,409,
        1,0,0,0,48,420,1,0,0,0,50,431,1,0,0,0,52,442,1,0,0,0,54,444,1,0,
        0,0,56,446,1,0,0,0,58,448,1,0,0,0,60,450,1,0,0,0,62,452,1,0,0,0,
        64,454,1,0,0,0,66,456,1,0,0,0,68,69,5,110,0,0,69,70,5,1,0,0,70,71,
        5,2,0,0,71,73,5,1,0,0,72,74,3,4,2,0,73,72,1,0,0,0,74,75,1,0,0,0,
        75,73,1,0,0,0,75,76,1,0,0,0,76,86,1,0,0,0,77,78,5,3,0,0,78,80,5,
        1,0,0,79,81,3,34,17,0,80,79,1,0,0,0,81,82,1,0,0,0,82,80,1,0,0,0,
        82,83,1,0,0,0,83,85,1,0,0,0,84,77,1,0,0,0,85,88,1,0,0,0,86,84,1,
        0,0,0,86,87,1,0,0,0,87,98,1,0,0,0,88,86,1,0,0,0,89,90,5,4,0,0,90,
        92,5,1,0,0,91,93,3,42,21,0,92,91,1,0,0,0,93,94,1,0,0,0,94,92,1,0,
        0,0,94,95,1,0,0,0,95,97,1,0,0,0,96,89,1,0,0,0,97,100,1,0,0,0,98,
        96,1,0,0,0,98,99,1,0,0,0,99,101,1,0,0,0,100,98,1,0,0,0,101,102,5,
        5,0,0,102,103,5,1,0,0,103,107,3,44,22,0,104,105,5,6,0,0,105,106,
        5,1,0,0,106,108,3,2,1,0,107,104,1,0,0,0,107,108,1,0,0,0,108,110,
        1,0,0,0,109,111,3,36,18,0,110,109,1,0,0,0,110,111,1,0,0,0,111,113,
        1,0,0,0,112,114,3,38,19,0,113,112,1,0,0,0,113,114,1,0,0,0,114,1,
        1,0,0,0,115,116,5,7,0,0,116,117,5,8,0,0,117,118,5,111,0,0,118,119,
        5,9,0,0,119,120,5,8,0,0,120,121,5,111,0,0,121,122,5,10,0,0,122,123,
        5,8,0,0,123,124,5,114,0,0,124,125,5,11,0,0,125,126,5,8,0,0,126,127,
        5,113,0,0,127,128,5,12,0,0,128,129,5,8,0,0,129,130,3,48,24,0,130,
        131,5,13,0,0,131,132,5,8,0,0,132,133,3,32,16,0,133,134,5,14,0,0,
        134,135,5,8,0,0,135,136,5,114,0,0,136,137,5,15,0,0,137,138,5,8,0,
        0,138,139,5,114,0,0,139,3,1,0,0,0,140,141,5,16,0,0,141,142,5,110,
        0,0,142,147,5,1,0,0,143,148,3,8,4,0,144,148,3,16,8,0,145,148,3,18,
        9,0,146,148,3,26,13,0,147,143,1,0,0,0,147,144,1,0,0,0,147,145,1,
        0,0,0,147,146,1,0,0,0,148,5,1,0,0,0,149,150,5,17,0,0,150,151,5,8,
        0,0,151,153,3,52,26,0,152,149,1,0,0,0,152,153,1,0,0,0,153,157,1,
        0,0,0,154,155,5,18,0,0,155,156,5,8,0,0,156,158,5,113,0,0,157,154,
        1,0,0,0,157,158,1,0,0,0,158,162,1,0,0,0,159,160,5,19,0,0,160,161,
        5,8,0,0,161,163,5,109,0,0,162,159,1,0,0,0,162,163,1,0,0,0,163,7,
        1,0,0,0,164,168,3,10,5,0,165,168,3,12,6,0,166,168,3,14,7,0,167,164,
        1,0,0,0,167,165,1,0,0,0,167,166,1,0,0,0,168,9,1,0,0,0,169,170,5,
        20,0,0,170,171,5,8,0,0,171,172,5,21,0,0,172,173,3,6,3,0,173,174,
        5,22,0,0,174,175,5,8,0,0,175,176,5,111,0,0,176,177,5,23,0,0,177,
        178,5,8,0,0,178,179,5,111,0,0,179,11,1,0,0,0,180,181,5,20,0,0,181,
        182,5,8,0,0,182,183,5,24,0,0,183,187,3,6,3,0,184,185,5,25,0,0,185,
        186,5,8,0,0,186,188,5,111,0,0,187,184,1,0,0,0,187,188,1,0,0,0,188,
        192,1,0,0,0,189,190,5,26,0,0,190,191,5,8,0,0,191,193,5,111,0,0,192,
        189,1,0,0,0,192,193,1,0,0,0,193,13,1,0,0,0,194,195,5,20,0,0,195,
        196,5,8,0,0,196,197,5,27,0,0,197,198,3,6,3,0,198,199,5,28,0,0,199,
        200,5,8,0,0,200,201,5,111,0,0,201,202,5,29,0,0,202,203,5,8,0,0,203,
        204,5,111,0,0,204,15,1,0,0,0,205,206,5,20,0,0,206,207,5,8,0,0,207,
        208,7,0,0,0,208,209,3,6,3,0,209,210,5,33,0,0,210,211,5,8,0,0,211,
        215,3,54,27,0,212,213,5,34,0,0,213,214,5,8,0,0,214,216,5,111,0,0,
        215,212,1,0,0,0,215,216,1,0,0,0,216,217,1,0,0,0,217,218,5,35,0,0,
        218,219,5,8,0,0,219,223,5,111,0,0,220,221,5,36,0,0,221,222,5,8,0,
        0,222,224,5,109,0,0,223,220,1,0,0,0,223,224,1,0,0,0,224,228,1,0,
        0,0,225,226,5,37,0,0,226,227,5,8,0,0,227,229,5,114,0,0,228,225,1,
        0,0,0,228,229,1,0,0,0,229,233,1,0,0,0,230,231,5,38,0,0,231,232,5,
        8,0,0,232,234,5,109,0,0,233,230,1,0,0,0,233,234,1,0,0,0,234,17,1,
        0,0,0,235,238,3,22,11,0,236,238,3,24,12,0,237,235,1,0,0,0,237,236,
        1,0,0,0,238,19,1,0,0,0,239,240,5,39,0,0,240,241,5,8,0,0,241,243,
        3,46,23,0,242,239,1,0,0,0,242,243,1,0,0,0,243,247,1,0,0,0,244,245,
        5,40,0,0,245,246,5,8,0,0,246,248,3,46,23,0,247,244,1,0,0,0,247,248,
        1,0,0,0,248,252,1,0,0,0,249,250,5,41,0,0,250,251,5,8,0,0,251,253,
        3,62,31,0,252,249,1,0,0,0,252,253,1,0,0,0,253,257,1,0,0,0,254,255,
        5,42,0,0,255,256,5,8,0,0,256,258,5,111,0,0,257,254,1,0,0,0,257,258,
        1,0,0,0,258,21,1,0,0,0,259,260,5,20,0,0,260,261,5,8,0,0,261,262,
        7,1,0,0,262,263,3,6,3,0,263,264,5,46,0,0,264,265,5,8,0,0,265,266,
        5,111,0,0,266,267,5,47,0,0,267,268,5,8,0,0,268,269,5,111,0,0,269,
        270,3,20,10,0,270,23,1,0,0,0,271,272,5,20,0,0,272,273,5,8,0,0,273,
        274,5,48,0,0,274,275,3,6,3,0,275,276,5,49,0,0,276,277,5,8,0,0,277,
        278,3,64,32,0,278,279,5,50,0,0,279,280,5,8,0,0,280,281,3,66,33,0,
        281,285,3,20,10,0,282,283,5,51,0,0,283,284,5,8,0,0,284,286,3,46,
        23,0,285,282,1,0,0,0,285,286,1,0,0,0,286,25,1,0,0,0,287,290,3,28,
        14,0,288,290,3,30,15,0,289,287,1,0,0,0,289,288,1,0,0,0,290,27,1,
        0,0,0,291,292,5,20,0,0,292,293,5,8,0,0,293,294,5,52,0,0,294,295,
        3,6,3,0,295,296,5,53,0,0,296,297,5,8,0,0,297,298,5,114,0,0,298,29,
        1,0,0,0,299,300,1,0,0,0,300,31,1,0,0,0,301,302,7,2,0,0,302,33,1,
        0,0,0,303,304,5,16,0,0,304,305,5,110,0,0,305,306,5,1,0,0,306,307,
        5,2,0,0,307,309,5,1,0,0,308,310,3,4,2,0,309,308,1,0,0,0,310,311,
        1,0,0,0,311,309,1,0,0,0,311,312,1,0,0,0,312,35,1,0,0,0,313,314,5,
        57,0,0,314,315,5,1,0,0,315,316,5,58,0,0,316,317,5,8,0,0,317,318,
        5,110,0,0,318,319,5,59,0,0,319,320,5,8,0,0,320,321,5,113,0,0,321,
        322,5,60,0,0,322,323,5,8,0,0,323,324,3,58,29,0,324,325,5,61,0,0,
        325,326,5,8,0,0,326,327,3,60,30,0,327,328,5,62,0,0,328,329,5,8,0,
        0,329,330,3,46,23,0,330,331,5,63,0,0,331,332,5,8,0,0,332,333,5,64,
        0,0,333,334,3,40,20,0,334,335,5,65,0,0,335,338,3,40,20,0,336,337,
        5,65,0,0,337,339,3,40,20,0,338,336,1,0,0,0,338,339,1,0,0,0,339,340,
        1,0,0,0,340,341,5,66,0,0,341,37,1,0,0,0,342,343,5,67,0,0,343,344,
        5,1,0,0,344,345,5,58,0,0,345,346,5,8,0,0,346,347,5,110,0,0,347,348,
        5,59,0,0,348,349,5,8,0,0,349,350,5,113,0,0,350,39,1,0,0,0,351,352,
        5,68,0,0,352,353,5,69,0,0,353,354,5,8,0,0,354,355,5,113,0,0,355,
        356,5,65,0,0,356,357,5,70,0,0,357,358,5,8,0,0,358,359,5,113,0,0,
        359,360,5,71,0,0,360,41,1,0,0,0,361,362,5,16,0,0,362,363,5,58,0,
        0,363,364,5,8,0,0,364,365,5,110,0,0,365,366,5,20,0,0,366,367,5,8,
        0,0,367,371,3,56,28,0,368,369,5,72,0,0,369,370,5,8,0,0,370,372,5,
        111,0,0,371,368,1,0,0,0,371,372,1,0,0,0,372,376,1,0,0,0,373,374,
        5,73,0,0,374,375,5,8,0,0,375,377,3,50,25,0,376,373,1,0,0,0,376,377,
        1,0,0,0,377,381,1,0,0,0,378,379,5,74,0,0,379,380,5,8,0,0,380,382,
        3,46,23,0,381,378,1,0,0,0,381,382,1,0,0,0,382,386,1,0,0,0,383,384,
        5,75,0,0,384,385,5,8,0,0,385,387,3,46,23,0,386,383,1,0,0,0,386,387,
        1,0,0,0,387,391,1,0,0,0,388,389,5,76,0,0,389,390,5,8,0,0,390,392,
        3,46,23,0,391,388,1,0,0,0,391,392,1,0,0,0,392,396,1,0,0,0,393,394,
        5,77,0,0,394,395,5,8,0,0,395,397,5,109,0,0,396,393,1,0,0,0,396,397,
        1,0,0,0,397,401,1,0,0,0,398,399,5,19,0,0,399,400,5,8,0,0,400,402,
        5,109,0,0,401,398,1,0,0,0,401,402,1,0,0,0,402,43,1,0,0,0,403,404,
        5,16,0,0,404,406,5,110,0,0,405,403,1,0,0,0,406,407,1,0,0,0,407,405,
        1,0,0,0,407,408,1,0,0,0,408,45,1,0,0,0,409,410,5,78,0,0,410,415,
        5,111,0,0,411,412,5,65,0,0,412,414,5,111,0,0,413,411,1,0,0,0,414,
        417,1,0,0,0,415,413,1,0,0,0,415,416,1,0,0,0,416,418,1,0,0,0,417,
        415,1,0,0,0,418,419,5,79,0,0,419,47,1,0,0,0,420,421,5,78,0,0,421,
        426,5,113,0,0,422,423,5,65,0,0,423,425,5,113,0,0,424,422,1,0,0,0,
        425,428,1,0,0,0,426,424,1,0,0,0,426,427,1,0,0,0,427,429,1,0,0,0,
        428,426,1,0,0,0,429,430,5,79,0,0,430,49,1,0,0,0,431,432,5,78,0,0,
        432,437,7,3,0,0,433,434,5,65,0,0,434,436,7,3,0,0,435,433,1,0,0,0,
        436,439,1,0,0,0,437,435,1,0,0,0,437,438,1,0,0,0,438,440,1,0,0,0,
        439,437,1,0,0,0,440,441,5,79,0,0,441,51,1,0,0,0,442,443,7,4,0,0,
        443,53,1,0,0,0,444,445,7,5,0,0,445,55,1,0,0,0,446,447,7,6,0,0,447,
        57,1,0,0,0,448,449,7,7,0,0,449,59,1,0,0,0,450,451,7,8,0,0,451,61,
        1,0,0,0,452,453,7,9,0,0,453,63,1,0,0,0,454,455,7,10,0,0,455,65,1,
        0,0,0,456,457,7,11,0,0,457,67,1,0,0,0,39,75,82,86,94,98,107,110,
        113,147,152,157,162,167,187,192,215,223,228,233,237,242,247,252,
        257,285,289,311,338,371,376,381,386,391,396,401,407,415,426,437
    ]

class NNParser ( Parser ):

    grammarFileName = "NN.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "':'", "'layers'", "'sub_nn'", "'tensor_ops'", 
                     "'modules'", "'config'", "'batch_size'", "'='", "'epochs'", 
                     "'learning_rate'", "'optimiser'", "'metrics'", "'loss_function'", 
                     "'weight_decay'", "'momentum'", "'-'", "'actv_func'", 
                     "'name_layer_input'", "'input_reused'", "'type'", "'Linear'", 
                     "'in_features'", "'out_features'", "'Flatten'", "'start_dim'", 
                     "'end_dim'", "'Embedding'", "'num_embeddings'", "'embedding_dim'", 
                     "'SimpleRNN'", "'LSTM'", "'GRU'", "'return_type'", 
                     "'input_size'", "'hidden_size'", "'bidirectional'", 
                     "'dropout'", "'batch_first'", "'kernel_dim'", "'stride_dim'", 
                     "'padding_type'", "'padding_amount'", "'Conv1D'", "'Conv2D'", 
                     "'Conv3D'", "'in_channels'", "'out_channels'", "'Pooling'", 
                     "'pooling_type'", "'dimension'", "'output_dim'", "'Dropout'", 
                     "'rate'", "'crossentropy'", "'binary_crossentropy'", 
                     "'mse'", "'TrainingDataset'", "'name'", "'path_data'", 
                     "'task_type'", "'input_format'", "'image'", "'labels'", 
                     "'{'", "','", "'}'", "'TestDataset'", "'('", "'col'", 
                     "'label'", "')'", "'concatenate_dim'", "'layers_of_tensors'", 
                     "'reshape_dim'", "'transpose_dim'", "'permute_dim'", 
                     "'after_activ_func'", "'['", "']'", "'relu'", "'leaky_relu'", 
                     "'sigmoid'", "'softmax'", "'tanh'", "'None'", "'last'", 
                     "'full'", "'hidden'", "'reshape'", "'concatenate'", 
                     "'multiply'", "'matmultiply'", "'permute'", "'transpose'", 
                     "'binary'", "'multi_class'", "'regression'", "'csv'", 
                     "'images'", "'valid'", "'same'", "'average'", "'max'", 
                     "'adaptive_average'", "'adaptive_max'", "'1D'", "'2D'", 
                     "'3D'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "BOOL", "ID", "INT", "WS", "STRING", 
                      "DOUBLE" ]

    RULE_neuralNetwork = 0
    RULE_parameters = 1
    RULE_layer = 2
    RULE_layerParams = 3
    RULE_generalLayer = 4
    RULE_linear = 5
    RULE_flatten = 6
    RULE_embedding = 7
    RULE_rnn = 8
    RULE_cnn = 9
    RULE_cnnParams = 10
    RULE_convolutional = 11
    RULE_pooling = 12
    RULE_layerModifier = 13
    RULE_dropout = 14
    RULE_normalisation = 15
    RULE_lossFunction = 16
    RULE_sub_nn = 17
    RULE_trainingDataset = 18
    RULE_testDataset = 19
    RULE_label = 20
    RULE_tensorOp = 21
    RULE_modules = 22
    RULE_intList = 23
    RULE_strList = 24
    RULE_intStrList = 25
    RULE_activityFuncType = 26
    RULE_returnTypeRRN = 27
    RULE_tensorOpType = 28
    RULE_taskType = 29
    RULE_inputFormat = 30
    RULE_paddingType = 31
    RULE_poolingType = 32
    RULE_dimensionality = 33

    ruleNames =  [ "neuralNetwork", "parameters", "layer", "layerParams", 
                   "generalLayer", "linear", "flatten", "embedding", "rnn", 
                   "cnn", "cnnParams", "convolutional", "pooling", "layerModifier", 
                   "dropout", "normalisation", "lossFunction", "sub_nn", 
                   "trainingDataset", "testDataset", "label", "tensorOp", 
                   "modules", "intList", "strList", "intStrList", "activityFuncType", 
                   "returnTypeRRN", "tensorOpType", "taskType", "inputFormat", 
                   "paddingType", "poolingType", "dimensionality" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    T__30=31
    T__31=32
    T__32=33
    T__33=34
    T__34=35
    T__35=36
    T__36=37
    T__37=38
    T__38=39
    T__39=40
    T__40=41
    T__41=42
    T__42=43
    T__43=44
    T__44=45
    T__45=46
    T__46=47
    T__47=48
    T__48=49
    T__49=50
    T__50=51
    T__51=52
    T__52=53
    T__53=54
    T__54=55
    T__55=56
    T__56=57
    T__57=58
    T__58=59
    T__59=60
    T__60=61
    T__61=62
    T__62=63
    T__63=64
    T__64=65
    T__65=66
    T__66=67
    T__67=68
    T__68=69
    T__69=70
    T__70=71
    T__71=72
    T__72=73
    T__73=74
    T__74=75
    T__75=76
    T__76=77
    T__77=78
    T__78=79
    T__79=80
    T__80=81
    T__81=82
    T__82=83
    T__83=84
    T__84=85
    T__85=86
    T__86=87
    T__87=88
    T__88=89
    T__89=90
    T__90=91
    T__91=92
    T__92=93
    T__93=94
    T__94=95
    T__95=96
    T__96=97
    T__97=98
    T__98=99
    T__99=100
    T__100=101
    T__101=102
    T__102=103
    T__103=104
    T__104=105
    T__105=106
    T__106=107
    T__107=108
    BOOL=109
    ID=110
    INT=111
    WS=112
    STRING=113
    DOUBLE=114

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class NeuralNetworkContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def modules(self):
            return self.getTypedRuleContext(NNParser.ModulesContext,0)


        def layer(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.LayerContext)
            else:
                return self.getTypedRuleContext(NNParser.LayerContext,i)


        def parameters(self):
            return self.getTypedRuleContext(NNParser.ParametersContext,0)


        def trainingDataset(self):
            return self.getTypedRuleContext(NNParser.TrainingDatasetContext,0)


        def testDataset(self):
            return self.getTypedRuleContext(NNParser.TestDatasetContext,0)


        def sub_nn(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.Sub_nnContext)
            else:
                return self.getTypedRuleContext(NNParser.Sub_nnContext,i)


        def tensorOp(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.TensorOpContext)
            else:
                return self.getTypedRuleContext(NNParser.TensorOpContext,i)


        def getRuleIndex(self):
            return NNParser.RULE_neuralNetwork

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNeuralNetwork" ):
                listener.enterNeuralNetwork(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNeuralNetwork" ):
                listener.exitNeuralNetwork(self)




    def neuralNetwork(self):

        localctx = NNParser.NeuralNetworkContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_neuralNetwork)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            self.match(NNParser.ID)
            self.state = 69
            self.match(NNParser.T__0)
            self.state = 70
            self.match(NNParser.T__1)
            self.state = 71
            self.match(NNParser.T__0)
            self.state = 73 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 72
                self.layer()
                self.state = 75 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==16):
                    break

            self.state = 86
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==3:
                self.state = 77
                self.match(NNParser.T__2)
                self.state = 78
                self.match(NNParser.T__0)
                self.state = 80 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 79
                    self.sub_nn()
                    self.state = 82 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==16):
                        break

                self.state = 88
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 98
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==4:
                self.state = 89
                self.match(NNParser.T__3)
                self.state = 90
                self.match(NNParser.T__0)
                self.state = 92 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 91
                    self.tensorOp()
                    self.state = 94 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==16):
                        break

                self.state = 100
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 101
            self.match(NNParser.T__4)
            self.state = 102
            self.match(NNParser.T__0)
            self.state = 103
            self.modules()
            self.state = 107
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==6:
                self.state = 104
                self.match(NNParser.T__5)
                self.state = 105
                self.match(NNParser.T__0)
                self.state = 106
                self.parameters()


            self.state = 110
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==57:
                self.state = 109
                self.trainingDataset()


            self.state = 113
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==67:
                self.state = 112
                self.testDataset()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ParametersContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def DOUBLE(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.DOUBLE)
            else:
                return self.getToken(NNParser.DOUBLE, i)

        def STRING(self):
            return self.getToken(NNParser.STRING, 0)

        def strList(self):
            return self.getTypedRuleContext(NNParser.StrListContext,0)


        def lossFunction(self):
            return self.getTypedRuleContext(NNParser.LossFunctionContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_parameters

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParameters" ):
                listener.enterParameters(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParameters" ):
                listener.exitParameters(self)




    def parameters(self):

        localctx = NNParser.ParametersContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_parameters)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 115
            self.match(NNParser.T__6)
            self.state = 116
            self.match(NNParser.T__7)
            self.state = 117
            self.match(NNParser.INT)
            self.state = 118
            self.match(NNParser.T__8)
            self.state = 119
            self.match(NNParser.T__7)
            self.state = 120
            self.match(NNParser.INT)
            self.state = 121
            self.match(NNParser.T__9)
            self.state = 122
            self.match(NNParser.T__7)
            self.state = 123
            self.match(NNParser.DOUBLE)
            self.state = 124
            self.match(NNParser.T__10)
            self.state = 125
            self.match(NNParser.T__7)
            self.state = 126
            self.match(NNParser.STRING)
            self.state = 127
            self.match(NNParser.T__11)
            self.state = 128
            self.match(NNParser.T__7)
            self.state = 129
            self.strList()
            self.state = 130
            self.match(NNParser.T__12)
            self.state = 131
            self.match(NNParser.T__7)
            self.state = 132
            self.lossFunction()
            self.state = 133
            self.match(NNParser.T__13)
            self.state = 134
            self.match(NNParser.T__7)
            self.state = 135
            self.match(NNParser.DOUBLE)
            self.state = 136
            self.match(NNParser.T__14)
            self.state = 137
            self.match(NNParser.T__7)
            self.state = 138
            self.match(NNParser.DOUBLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LayerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def generalLayer(self):
            return self.getTypedRuleContext(NNParser.GeneralLayerContext,0)


        def rnn(self):
            return self.getTypedRuleContext(NNParser.RnnContext,0)


        def cnn(self):
            return self.getTypedRuleContext(NNParser.CnnContext,0)


        def layerModifier(self):
            return self.getTypedRuleContext(NNParser.LayerModifierContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_layer

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLayer" ):
                listener.enterLayer(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLayer" ):
                listener.exitLayer(self)




    def layer(self):

        localctx = NNParser.LayerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_layer)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 140
            self.match(NNParser.T__15)
            self.state = 141
            self.match(NNParser.ID)
            self.state = 142
            self.match(NNParser.T__0)
            self.state = 147
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
            if la_ == 1:
                self.state = 143
                self.generalLayer()
                pass

            elif la_ == 2:
                self.state = 144
                self.rnn()
                pass

            elif la_ == 3:
                self.state = 145
                self.cnn()
                pass

            elif la_ == 4:
                self.state = 146
                self.layerModifier()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LayerParamsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def activityFuncType(self):
            return self.getTypedRuleContext(NNParser.ActivityFuncTypeContext,0)


        def STRING(self):
            return self.getToken(NNParser.STRING, 0)

        def BOOL(self):
            return self.getToken(NNParser.BOOL, 0)

        def getRuleIndex(self):
            return NNParser.RULE_layerParams

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLayerParams" ):
                listener.enterLayerParams(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLayerParams" ):
                listener.exitLayerParams(self)




    def layerParams(self):

        localctx = NNParser.LayerParamsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_layerParams)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 152
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==17:
                self.state = 149
                self.match(NNParser.T__16)
                self.state = 150
                self.match(NNParser.T__7)
                self.state = 151
                self.activityFuncType()


            self.state = 157
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==18:
                self.state = 154
                self.match(NNParser.T__17)
                self.state = 155
                self.match(NNParser.T__7)
                self.state = 156
                self.match(NNParser.STRING)


            self.state = 162
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==19:
                self.state = 159
                self.match(NNParser.T__18)
                self.state = 160
                self.match(NNParser.T__7)
                self.state = 161
                self.match(NNParser.BOOL)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GeneralLayerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def linear(self):
            return self.getTypedRuleContext(NNParser.LinearContext,0)


        def flatten(self):
            return self.getTypedRuleContext(NNParser.FlattenContext,0)


        def embedding(self):
            return self.getTypedRuleContext(NNParser.EmbeddingContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_generalLayer

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGeneralLayer" ):
                listener.enterGeneralLayer(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGeneralLayer" ):
                listener.exitGeneralLayer(self)




    def generalLayer(self):

        localctx = NNParser.GeneralLayerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_generalLayer)
        try:
            self.state = 167
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 164
                self.linear()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 165
                self.flatten()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 166
                self.embedding()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LinearContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def getRuleIndex(self):
            return NNParser.RULE_linear

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLinear" ):
                listener.enterLinear(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLinear" ):
                listener.exitLinear(self)




    def linear(self):

        localctx = NNParser.LinearContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_linear)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 169
            self.match(NNParser.T__19)
            self.state = 170
            self.match(NNParser.T__7)
            self.state = 171
            self.match(NNParser.T__20)
            self.state = 172
            self.layerParams()
            self.state = 173
            self.match(NNParser.T__21)
            self.state = 174
            self.match(NNParser.T__7)
            self.state = 175
            self.match(NNParser.INT)
            self.state = 176
            self.match(NNParser.T__22)
            self.state = 177
            self.match(NNParser.T__7)
            self.state = 178
            self.match(NNParser.INT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FlattenContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def getRuleIndex(self):
            return NNParser.RULE_flatten

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFlatten" ):
                listener.enterFlatten(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFlatten" ):
                listener.exitFlatten(self)




    def flatten(self):

        localctx = NNParser.FlattenContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_flatten)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 180
            self.match(NNParser.T__19)
            self.state = 181
            self.match(NNParser.T__7)
            self.state = 182
            self.match(NNParser.T__23)
            self.state = 183
            self.layerParams()
            self.state = 187
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==25:
                self.state = 184
                self.match(NNParser.T__24)
                self.state = 185
                self.match(NNParser.T__7)
                self.state = 186
                self.match(NNParser.INT)


            self.state = 192
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==26:
                self.state = 189
                self.match(NNParser.T__25)
                self.state = 190
                self.match(NNParser.T__7)
                self.state = 191
                self.match(NNParser.INT)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EmbeddingContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def getRuleIndex(self):
            return NNParser.RULE_embedding

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEmbedding" ):
                listener.enterEmbedding(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEmbedding" ):
                listener.exitEmbedding(self)




    def embedding(self):

        localctx = NNParser.EmbeddingContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_embedding)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 194
            self.match(NNParser.T__19)
            self.state = 195
            self.match(NNParser.T__7)
            self.state = 196
            self.match(NNParser.T__26)
            self.state = 197
            self.layerParams()
            self.state = 198
            self.match(NNParser.T__27)
            self.state = 199
            self.match(NNParser.T__7)
            self.state = 200
            self.match(NNParser.INT)
            self.state = 201
            self.match(NNParser.T__28)
            self.state = 202
            self.match(NNParser.T__7)
            self.state = 203
            self.match(NNParser.INT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RnnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.rnn_type = None # Token
            self.i_size = None # Token
            self.bid = None # Token
            self.dout = None # Token
            self.b_first = None # Token

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def returnTypeRRN(self):
            return self.getTypedRuleContext(NNParser.ReturnTypeRRNContext,0)


        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def BOOL(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.BOOL)
            else:
                return self.getToken(NNParser.BOOL, i)

        def DOUBLE(self):
            return self.getToken(NNParser.DOUBLE, 0)

        def getRuleIndex(self):
            return NNParser.RULE_rnn

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRnn" ):
                listener.enterRnn(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRnn" ):
                listener.exitRnn(self)




    def rnn(self):

        localctx = NNParser.RnnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_rnn)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 205
            self.match(NNParser.T__19)
            self.state = 206
            self.match(NNParser.T__7)
            self.state = 207
            localctx.rnn_type = self._input.LT(1)
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 7516192768) != 0)):
                localctx.rnn_type = self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 208
            self.layerParams()
            self.state = 209
            self.match(NNParser.T__32)
            self.state = 210
            self.match(NNParser.T__7)
            self.state = 211
            self.returnTypeRRN()
            self.state = 215
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==34:
                self.state = 212
                self.match(NNParser.T__33)
                self.state = 213
                self.match(NNParser.T__7)
                self.state = 214
                localctx.i_size = self.match(NNParser.INT)


            self.state = 217
            self.match(NNParser.T__34)
            self.state = 218
            self.match(NNParser.T__7)
            self.state = 219
            self.match(NNParser.INT)
            self.state = 223
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==36:
                self.state = 220
                self.match(NNParser.T__35)
                self.state = 221
                self.match(NNParser.T__7)
                self.state = 222
                localctx.bid = self.match(NNParser.BOOL)


            self.state = 228
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==37:
                self.state = 225
                self.match(NNParser.T__36)
                self.state = 226
                self.match(NNParser.T__7)
                self.state = 227
                localctx.dout = self.match(NNParser.DOUBLE)


            self.state = 233
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==38:
                self.state = 230
                self.match(NNParser.T__37)
                self.state = 231
                self.match(NNParser.T__7)
                self.state = 232
                localctx.b_first = self.match(NNParser.BOOL)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CnnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def convolutional(self):
            return self.getTypedRuleContext(NNParser.ConvolutionalContext,0)


        def pooling(self):
            return self.getTypedRuleContext(NNParser.PoolingContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_cnn

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCnn" ):
                listener.enterCnn(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCnn" ):
                listener.exitCnn(self)




    def cnn(self):

        localctx = NNParser.CnnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_cnn)
        try:
            self.state = 237
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 235
                self.convolutional()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 236
                self.pooling()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CnnParamsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.kernel = None # IntListContext
            self.stride = None # IntListContext

        def paddingType(self):
            return self.getTypedRuleContext(NNParser.PaddingTypeContext,0)


        def INT(self):
            return self.getToken(NNParser.INT, 0)

        def intList(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.IntListContext)
            else:
                return self.getTypedRuleContext(NNParser.IntListContext,i)


        def getRuleIndex(self):
            return NNParser.RULE_cnnParams

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCnnParams" ):
                listener.enterCnnParams(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCnnParams" ):
                listener.exitCnnParams(self)




    def cnnParams(self):

        localctx = NNParser.CnnParamsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_cnnParams)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 242
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==39:
                self.state = 239
                self.match(NNParser.T__38)
                self.state = 240
                self.match(NNParser.T__7)
                self.state = 241
                localctx.kernel = self.intList()


            self.state = 247
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==40:
                self.state = 244
                self.match(NNParser.T__39)
                self.state = 245
                self.match(NNParser.T__7)
                self.state = 246
                localctx.stride = self.intList()


            self.state = 252
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==41:
                self.state = 249
                self.match(NNParser.T__40)
                self.state = 250
                self.match(NNParser.T__7)
                self.state = 251
                self.paddingType()


            self.state = 257
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==42:
                self.state = 254
                self.match(NNParser.T__41)
                self.state = 255
                self.match(NNParser.T__7)
                self.state = 256
                self.match(NNParser.INT)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConvolutionalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.conv_type = None # Token

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def cnnParams(self):
            return self.getTypedRuleContext(NNParser.CnnParamsContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_convolutional

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConvolutional" ):
                listener.enterConvolutional(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConvolutional" ):
                listener.exitConvolutional(self)




    def convolutional(self):

        localctx = NNParser.ConvolutionalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_convolutional)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 259
            self.match(NNParser.T__19)
            self.state = 260
            self.match(NNParser.T__7)
            self.state = 261
            localctx.conv_type = self._input.LT(1)
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 61572651155456) != 0)):
                localctx.conv_type = self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 262
            self.layerParams()
            self.state = 263
            self.match(NNParser.T__45)
            self.state = 264
            self.match(NNParser.T__7)
            self.state = 265
            self.match(NNParser.INT)
            self.state = 266
            self.match(NNParser.T__46)
            self.state = 267
            self.match(NNParser.T__7)
            self.state = 268
            self.match(NNParser.INT)
            self.state = 269
            self.cnnParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PoolingContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def poolingType(self):
            return self.getTypedRuleContext(NNParser.PoolingTypeContext,0)


        def dimensionality(self):
            return self.getTypedRuleContext(NNParser.DimensionalityContext,0)


        def cnnParams(self):
            return self.getTypedRuleContext(NNParser.CnnParamsContext,0)


        def intList(self):
            return self.getTypedRuleContext(NNParser.IntListContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_pooling

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPooling" ):
                listener.enterPooling(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPooling" ):
                listener.exitPooling(self)




    def pooling(self):

        localctx = NNParser.PoolingContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_pooling)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 271
            self.match(NNParser.T__19)
            self.state = 272
            self.match(NNParser.T__7)
            self.state = 273
            self.match(NNParser.T__47)
            self.state = 274
            self.layerParams()
            self.state = 275
            self.match(NNParser.T__48)
            self.state = 276
            self.match(NNParser.T__7)
            self.state = 277
            self.poolingType()
            self.state = 278
            self.match(NNParser.T__49)
            self.state = 279
            self.match(NNParser.T__7)
            self.state = 280
            self.dimensionality()
            self.state = 281
            self.cnnParams()
            self.state = 285
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==51:
                self.state = 282
                self.match(NNParser.T__50)
                self.state = 283
                self.match(NNParser.T__7)
                self.state = 284
                self.intList()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LayerModifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def dropout(self):
            return self.getTypedRuleContext(NNParser.DropoutContext,0)


        def normalisation(self):
            return self.getTypedRuleContext(NNParser.NormalisationContext,0)


        def getRuleIndex(self):
            return NNParser.RULE_layerModifier

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLayerModifier" ):
                listener.enterLayerModifier(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLayerModifier" ):
                listener.exitLayerModifier(self)




    def layerModifier(self):

        localctx = NNParser.LayerModifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_layerModifier)
        try:
            self.state = 289
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [20]:
                self.enterOuterAlt(localctx, 1)
                self.state = 287
                self.dropout()
                pass
            elif token in [3, 4, 5, 16]:
                self.enterOuterAlt(localctx, 2)
                self.state = 288
                self.normalisation()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DropoutContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def layerParams(self):
            return self.getTypedRuleContext(NNParser.LayerParamsContext,0)


        def DOUBLE(self):
            return self.getToken(NNParser.DOUBLE, 0)

        def getRuleIndex(self):
            return NNParser.RULE_dropout

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDropout" ):
                listener.enterDropout(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDropout" ):
                listener.exitDropout(self)




    def dropout(self):

        localctx = NNParser.DropoutContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_dropout)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 291
            self.match(NNParser.T__19)
            self.state = 292
            self.match(NNParser.T__7)
            self.state = 293
            self.match(NNParser.T__51)
            self.state = 294
            self.layerParams()
            self.state = 295
            self.match(NNParser.T__52)
            self.state = 296
            self.match(NNParser.T__7)
            self.state = 297
            self.match(NNParser.DOUBLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class NormalisationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_normalisation

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNormalisation" ):
                listener.enterNormalisation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNormalisation" ):
                listener.exitNormalisation(self)




    def normalisation(self):

        localctx = NNParser.NormalisationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_normalisation)
        try:
            self.enterOuterAlt(localctx, 1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LossFunctionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_lossFunction

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLossFunction" ):
                listener.enterLossFunction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLossFunction" ):
                listener.exitLossFunction(self)




    def lossFunction(self):

        localctx = NNParser.LossFunctionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_lossFunction)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 301
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 126100789566373888) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Sub_nnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def layer(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.LayerContext)
            else:
                return self.getTypedRuleContext(NNParser.LayerContext,i)


        def getRuleIndex(self):
            return NNParser.RULE_sub_nn

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSub_nn" ):
                listener.enterSub_nn(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSub_nn" ):
                listener.exitSub_nn(self)




    def sub_nn(self):

        localctx = NNParser.Sub_nnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_sub_nn)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 303
            self.match(NNParser.T__15)
            self.state = 304
            self.match(NNParser.ID)
            self.state = 305
            self.match(NNParser.T__0)
            self.state = 306
            self.match(NNParser.T__1)
            self.state = 307
            self.match(NNParser.T__0)
            self.state = 309 
            self._errHandler.sync(self)
            _alt = 1
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 308
                    self.layer()

                else:
                    raise NoViableAltException(self)
                self.state = 311 
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,26,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TrainingDatasetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def STRING(self):
            return self.getToken(NNParser.STRING, 0)

        def taskType(self):
            return self.getTypedRuleContext(NNParser.TaskTypeContext,0)


        def inputFormat(self):
            return self.getTypedRuleContext(NNParser.InputFormatContext,0)


        def intList(self):
            return self.getTypedRuleContext(NNParser.IntListContext,0)


        def label(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.LabelContext)
            else:
                return self.getTypedRuleContext(NNParser.LabelContext,i)


        def getRuleIndex(self):
            return NNParser.RULE_trainingDataset

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrainingDataset" ):
                listener.enterTrainingDataset(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrainingDataset" ):
                listener.exitTrainingDataset(self)




    def trainingDataset(self):

        localctx = NNParser.TrainingDatasetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_trainingDataset)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 313
            self.match(NNParser.T__56)
            self.state = 314
            self.match(NNParser.T__0)
            self.state = 315
            self.match(NNParser.T__57)
            self.state = 316
            self.match(NNParser.T__7)
            self.state = 317
            self.match(NNParser.ID)
            self.state = 318
            self.match(NNParser.T__58)
            self.state = 319
            self.match(NNParser.T__7)
            self.state = 320
            self.match(NNParser.STRING)
            self.state = 321
            self.match(NNParser.T__59)
            self.state = 322
            self.match(NNParser.T__7)
            self.state = 323
            self.taskType()
            self.state = 324
            self.match(NNParser.T__60)
            self.state = 325
            self.match(NNParser.T__7)
            self.state = 326
            self.inputFormat()
            self.state = 327
            self.match(NNParser.T__61)
            self.state = 328
            self.match(NNParser.T__7)
            self.state = 329
            self.intList()
            self.state = 330
            self.match(NNParser.T__62)
            self.state = 331
            self.match(NNParser.T__7)
            self.state = 332
            self.match(NNParser.T__63)
            self.state = 333
            self.label()
            self.state = 334
            self.match(NNParser.T__64)
            self.state = 335
            self.label()
            self.state = 338
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==65:
                self.state = 336
                self.match(NNParser.T__64)
                self.state = 337
                self.label()


            self.state = 340
            self.match(NNParser.T__65)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TestDatasetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def STRING(self):
            return self.getToken(NNParser.STRING, 0)

        def getRuleIndex(self):
            return NNParser.RULE_testDataset

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTestDataset" ):
                listener.enterTestDataset(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTestDataset" ):
                listener.exitTestDataset(self)




    def testDataset(self):

        localctx = NNParser.TestDatasetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_testDataset)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 342
            self.match(NNParser.T__66)
            self.state = 343
            self.match(NNParser.T__0)
            self.state = 344
            self.match(NNParser.T__57)
            self.state = 345
            self.match(NNParser.T__7)
            self.state = 346
            self.match(NNParser.ID)
            self.state = 347
            self.match(NNParser.T__58)
            self.state = 348
            self.match(NNParser.T__7)
            self.state = 349
            self.match(NNParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LabelContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.STRING)
            else:
                return self.getToken(NNParser.STRING, i)

        def getRuleIndex(self):
            return NNParser.RULE_label

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLabel" ):
                listener.enterLabel(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLabel" ):
                listener.exitLabel(self)




    def label(self):

        localctx = NNParser.LabelContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_label)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 351
            self.match(NNParser.T__67)
            self.state = 352
            self.match(NNParser.T__68)
            self.state = 353
            self.match(NNParser.T__7)
            self.state = 354
            self.match(NNParser.STRING)
            self.state = 355
            self.match(NNParser.T__64)
            self.state = 356
            self.match(NNParser.T__69)
            self.state = 357
            self.match(NNParser.T__7)
            self.state = 358
            self.match(NNParser.STRING)
            self.state = 359
            self.match(NNParser.T__70)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TensorOpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.reshape = None # IntListContext
            self.transpose = None # IntListContext
            self.permute = None # IntListContext
            self.after_ativ = None # Token
            self.input_ref = None # Token

        def ID(self):
            return self.getToken(NNParser.ID, 0)

        def tensorOpType(self):
            return self.getTypedRuleContext(NNParser.TensorOpTypeContext,0)


        def INT(self):
            return self.getToken(NNParser.INT, 0)

        def intStrList(self):
            return self.getTypedRuleContext(NNParser.IntStrListContext,0)


        def intList(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NNParser.IntListContext)
            else:
                return self.getTypedRuleContext(NNParser.IntListContext,i)


        def BOOL(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.BOOL)
            else:
                return self.getToken(NNParser.BOOL, i)

        def getRuleIndex(self):
            return NNParser.RULE_tensorOp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTensorOp" ):
                listener.enterTensorOp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTensorOp" ):
                listener.exitTensorOp(self)




    def tensorOp(self):

        localctx = NNParser.TensorOpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_tensorOp)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 361
            self.match(NNParser.T__15)
            self.state = 362
            self.match(NNParser.T__57)
            self.state = 363
            self.match(NNParser.T__7)
            self.state = 364
            self.match(NNParser.ID)
            self.state = 365
            self.match(NNParser.T__19)
            self.state = 366
            self.match(NNParser.T__7)
            self.state = 367
            self.tensorOpType()
            self.state = 371
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==72:
                self.state = 368
                self.match(NNParser.T__71)
                self.state = 369
                self.match(NNParser.T__7)
                self.state = 370
                self.match(NNParser.INT)


            self.state = 376
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==73:
                self.state = 373
                self.match(NNParser.T__72)
                self.state = 374
                self.match(NNParser.T__7)
                self.state = 375
                self.intStrList()


            self.state = 381
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==74:
                self.state = 378
                self.match(NNParser.T__73)
                self.state = 379
                self.match(NNParser.T__7)
                self.state = 380
                localctx.reshape = self.intList()


            self.state = 386
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==75:
                self.state = 383
                self.match(NNParser.T__74)
                self.state = 384
                self.match(NNParser.T__7)
                self.state = 385
                localctx.transpose = self.intList()


            self.state = 391
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==76:
                self.state = 388
                self.match(NNParser.T__75)
                self.state = 389
                self.match(NNParser.T__7)
                self.state = 390
                localctx.permute = self.intList()


            self.state = 396
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==77:
                self.state = 393
                self.match(NNParser.T__76)
                self.state = 394
                self.match(NNParser.T__7)
                self.state = 395
                localctx.after_ativ = self.match(NNParser.BOOL)


            self.state = 401
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==19:
                self.state = 398
                self.match(NNParser.T__18)
                self.state = 399
                self.match(NNParser.T__7)
                self.state = 400
                localctx.input_ref = self.match(NNParser.BOOL)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ModulesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.ID)
            else:
                return self.getToken(NNParser.ID, i)

        def getRuleIndex(self):
            return NNParser.RULE_modules

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModules" ):
                listener.enterModules(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModules" ):
                listener.exitModules(self)




    def modules(self):

        localctx = NNParser.ModulesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_modules)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 405 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 403
                self.match(NNParser.T__15)
                self.state = 404
                self.match(NNParser.ID)
                self.state = 407 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==16):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IntListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def getRuleIndex(self):
            return NNParser.RULE_intList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIntList" ):
                listener.enterIntList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIntList" ):
                listener.exitIntList(self)




    def intList(self):

        localctx = NNParser.IntListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_intList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 409
            self.match(NNParser.T__77)
            self.state = 410
            self.match(NNParser.INT)
            self.state = 415
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==65:
                self.state = 411
                self.match(NNParser.T__64)
                self.state = 412
                self.match(NNParser.INT)
                self.state = 417
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 418
            self.match(NNParser.T__78)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StrListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.STRING)
            else:
                return self.getToken(NNParser.STRING, i)

        def getRuleIndex(self):
            return NNParser.RULE_strList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStrList" ):
                listener.enterStrList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStrList" ):
                listener.exitStrList(self)




    def strList(self):

        localctx = NNParser.StrListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_strList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 420
            self.match(NNParser.T__77)
            self.state = 421
            self.match(NNParser.STRING)
            self.state = 426
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==65:
                self.state = 422
                self.match(NNParser.T__64)
                self.state = 423
                self.match(NNParser.STRING)
                self.state = 428
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 429
            self.match(NNParser.T__78)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IntStrListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.INT)
            else:
                return self.getToken(NNParser.INT, i)

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.STRING)
            else:
                return self.getToken(NNParser.STRING, i)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(NNParser.ID)
            else:
                return self.getToken(NNParser.ID, i)

        def getRuleIndex(self):
            return NNParser.RULE_intStrList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIntStrList" ):
                listener.enterIntStrList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIntStrList" ):
                listener.exitIntStrList(self)




    def intStrList(self):

        localctx = NNParser.IntStrListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_intStrList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 431
            self.match(NNParser.T__77)
            self.state = 432
            _la = self._input.LA(1)
            if not(((((_la - 110)) & ~0x3f) == 0 and ((1 << (_la - 110)) & 11) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 437
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==65:
                self.state = 433
                self.match(NNParser.T__64)
                self.state = 434
                _la = self._input.LA(1)
                if not(((((_la - 110)) & ~0x3f) == 0 and ((1 << (_la - 110)) & 11) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 439
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 440
            self.match(NNParser.T__78)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ActivityFuncTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_activityFuncType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterActivityFuncType" ):
                listener.enterActivityFuncType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitActivityFuncType" ):
                listener.exitActivityFuncType(self)




    def activityFuncType(self):

        localctx = NNParser.ActivityFuncTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_activityFuncType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 442
            _la = self._input.LA(1)
            if not(((((_la - 80)) & ~0x3f) == 0 and ((1 << (_la - 80)) & 63) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ReturnTypeRRNContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_returnTypeRRN

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterReturnTypeRRN" ):
                listener.enterReturnTypeRRN(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitReturnTypeRRN" ):
                listener.exitReturnTypeRRN(self)




    def returnTypeRRN(self):

        localctx = NNParser.ReturnTypeRRNContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_returnTypeRRN)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 444
            _la = self._input.LA(1)
            if not(((((_la - 86)) & ~0x3f) == 0 and ((1 << (_la - 86)) & 7) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TensorOpTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_tensorOpType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTensorOpType" ):
                listener.enterTensorOpType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTensorOpType" ):
                listener.exitTensorOpType(self)




    def tensorOpType(self):

        localctx = NNParser.TensorOpTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_tensorOpType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 446
            _la = self._input.LA(1)
            if not(((((_la - 89)) & ~0x3f) == 0 and ((1 << (_la - 89)) & 63) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TaskTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_taskType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTaskType" ):
                listener.enterTaskType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTaskType" ):
                listener.exitTaskType(self)




    def taskType(self):

        localctx = NNParser.TaskTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_taskType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 448
            _la = self._input.LA(1)
            if not(((((_la - 95)) & ~0x3f) == 0 and ((1 << (_la - 95)) & 7) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InputFormatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_inputFormat

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInputFormat" ):
                listener.enterInputFormat(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInputFormat" ):
                listener.exitInputFormat(self)




    def inputFormat(self):

        localctx = NNParser.InputFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_inputFormat)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 450
            _la = self._input.LA(1)
            if not(_la==98 or _la==99):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PaddingTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_paddingType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPaddingType" ):
                listener.enterPaddingType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPaddingType" ):
                listener.exitPaddingType(self)




    def paddingType(self):

        localctx = NNParser.PaddingTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_paddingType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 452
            _la = self._input.LA(1)
            if not(_la==100 or _la==101):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PoolingTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_poolingType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPoolingType" ):
                listener.enterPoolingType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPoolingType" ):
                listener.exitPoolingType(self)




    def poolingType(self):

        localctx = NNParser.PoolingTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 64, self.RULE_poolingType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 454
            _la = self._input.LA(1)
            if not(((((_la - 102)) & ~0x3f) == 0 and ((1 << (_la - 102)) & 15) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DimensionalityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return NNParser.RULE_dimensionality

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDimensionality" ):
                listener.enterDimensionality(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDimensionality" ):
                listener.exitDimensionality(self)




    def dimensionality(self):

        localctx = NNParser.DimensionalityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_dimensionality)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 456
            _la = self._input.LA(1)
            if not(((((_la - 106)) & ~0x3f) == 0 and ((1 << (_la - 106)) & 7) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





