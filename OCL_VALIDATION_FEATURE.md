# OCL Constraint Validation - Feature Summary

## Overview

This feature adds automatic OCL (Object Constraint Language) constraint validation to the BESSER code generation pipeline. When you define OCL invariant constraints in your B-UML model, they are automatically:

1. **Parsed** from the OCL expression using BESSER's ANTLR-based OCL parser
2. **Transformed** into Pydantic field validators in the generated backend code
3. **Displayed** as user-friendly error messages in the React frontend when validation fails

## What Changed

### Backend - Pydantic Generator

| File | Change |
|------|--------|
| `besser/generators/pydantic_classes/ocl_utils.py` | **NEW** - OCL parser utilities |
| `besser/generators/pydantic_classes/pydantic_classes_generator.py` | Passes constraints to template |
| `templates/pydantic_classes_template.py.j2` | Generates `@field_validator` decorators |

### Frontend - React Generator

| File | Change |
|------|--------|
| `templates/src/components/table/TableComponent.tsx.j2` | Shows validation errors in form modal |
| `templates/src/components/MethodButton.tsx.j2` | Improved 500 error handling, keeps modal open on error |

## Supported OCL Constraints

Currently supports **simple invariant constraints** of the form:

```ocl
context Player inv minAge: self.age > 10
context Player inv maxJersey: self.jerseyNumber <= 99
context Team inv nonEmptyName: self.name <> ''
```

### Supported Operators

| OCL Operator | Python Equivalent | Example |
|--------------|-------------------|---------|
| `>` | `>` | `self.age > 18` |
| `<` | `<` | `self.age < 65` |
| `>=` | `>=` | `self.score >= 0` |
| `<=` | `<=` | `self.price <= 100` |
| `=` | `==` | `self.status = 'active'` |
| `<>` | `!=` | `self.name <> ''` |

### Supported Value Types

- Integer: `10`, `-5`
- Float: `3.14`, `100.0`
- String: `'hello'`, `''`
- Boolean: `True`, `False`

## Generated Code Example

### Input (B-UML Model)

```python
from besser.BUML.metamodel.structural import Constraint

constraint = Constraint(
    name="min_age",
    context=Player,
    expression="context Player inv: self.age > 10",
    language="OCL"
)
domain_model.constraints = {constraint}
```

### Output (Generated Pydantic)

```python
from pydantic import BaseModel, field_validator

class PlayerCreate(BaseModel):
    age: int
    name: str
    
    @field_validator('age')
    @classmethod
    def validate_age_1(cls, v):
        """OCL Constraint: min_age"""
        if not (v > 10):
            raise ValueError('age must be > 10')
        return v
```

## Frontend Error Display

### Form Validation (TableComponent)

When a user submits a form with invalid data:
1. Backend returns HTTP 422 with validation error
2. Frontend parses the error response
3. Error displayed in red box inside the modal
4. Modal stays open for user to fix and retry

### Method Execution (MethodButton)

When a method fails with HTTP 500:
1. Error message extracted from response
2. For methods with parameters: error shown inside modal
3. For methods without parameters: error toast appears
4. User can retry without re-entering data

## Test Suite

50 tests covering the OCL validation feature:

```bash
python -m pytest tests/generators/backend/test_ocl_utils.py tests/generators/backend/test_ocl_pydantic_integration.py -v
```

## Usage

```python
# Define constraints in your model
from besser.BUML.metamodel.structural import Constraint

constraint = Constraint(
    name="age_constraint",
    context=Player,
    expression="context Player inv: self.age > 10",
    language="OCL"
)
domain_model.constraints = {constraint}

# Generate backend with validators
from besser.generators.pydantic_classes import PydanticGenerator
gen = PydanticGenerator(model=domain_model, backend=True)
gen.generate()

# Or generate full web app
from besser.generators.web_app import WebAppGenerator
gen = WebAppGenerator(domain_model, gui_model, output_dir="output")
gen.generate()
```

## Future Enhancements

Potential extensions for this feature:
- Support for multi-property constraints (`self.endDate > self.startDate`)
- Collection operations (`self.items->size() > 0`)
- String operations (`self.name.size() > 3`)
- SQLAlchemy CHECK constraint generation
