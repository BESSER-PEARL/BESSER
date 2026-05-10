# API Configuration
API_VERSION = "1.0.0"

# Temp directory prefixes
TEMP_DIR_PREFIX = "besser_"
AGENT_TEMP_DIR_PREFIX = "besser_agent_"
CSV_TEMP_DIR_PREFIX = "besser_csv_"
DB_TEMP_DIR_PREFIX = "besser_db_"

# Output defaults
OUTPUT_DIR_NAME = "output"
AGENT_MODEL_FILENAME = "agent_model.py"
AGENT_OUTPUT_FILENAME = "agent_output.zip"

# Generator defaults
DEFAULT_SQL_DIALECT = "standard"
DEFAULT_DBMS = "sqlite"
DEFAULT_JSONSCHEMA_MODE = "regular"
DEFAULT_QISKIT_BACKEND = "aer_simulator"
DEFAULT_QISKIT_SHOTS = 1024
DEFAULT_DJANGO_PROJECT_NAME = "myproject"
DEFAULT_DJANGO_APP_NAME = "myapp"

# CORS defaults
DEFAULT_CORS_ORIGINS = [
    "https://editor.besser-pearl.org",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
]

VALID_PRIMITIVE_TYPES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "timedelta": "timedelta",
    "any": "any"
}

VISIBILITY_MAP = {
    "+": "public",
    "-": "private",
    "#": "protected",
    "~": "package"
}

RELATIONSHIP_TYPES = {
    "bidirectional": "ClassBidirectional",
    "unidirectional": "ClassUnidirectional",
    "composition": "ClassComposition",
    "aggregation": "ClassAggregation",
    "inheritance": "ClassInheritance"
}
