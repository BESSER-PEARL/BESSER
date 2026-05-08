# A7 — Image-to-ClassDiagram Endpoint Analysis

**Wave**: BESSER WME final-analysis
**Scope**: Verify `POST /get-json-model-from-image` produces valid v4 ClassDiagram output (was v3 before SA-6.1).
**Verdict**: **CLEAN — v4 native, no conversion at the boundary.**

---

## 1. Endpoint Location

`besser/utilities/web_modeling_editor/backend/routers/conversion_router.py:626-687`

```python
@router.post("/get-json-model-from-image", response_model=DiagramExportResponse)
@handle_endpoint_errors("get_json_model_from_image")
async def get_json_model_from_image(
    image_file: UploadFile = File(...),
    api_key: str = Form(...),
    existing_model: str = Form(None),
):
    ...
    domain_model = image_to_buml(
        image_path=image_path,
        openai_token=api_key,
        existing_model=existing_domain_model,
    )
    diagram_json = class_buml_to_json(domain_model)
    ...
    return {
        "title": diagram_title,
        "model": {**diagram_json, "type": diagram_type},
        "diagramType": "ClassDiagram",
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
    }
```

## 2. Pipeline / Output Shape

```
PNG/JPEG upload
   │
   ▼
image_to_buml()             besser/utilities/image_to_buml.py:150
   │  ├── image_to_plantuml() — OpenAI prompt asks for PlantUML
   │  │     (NOT v3 JSON, NOT v4 JSON — PlantUML text)
   │  └── plantuml_to_buml() — ANTLR parser → DomainModel
   ▼
DomainModel (B-UML metamodel object)
   │
   ▼
class_buml_to_json(domain_model)
   ▼
v4 ClassDiagram JSON
   { "version": "4.0.0",
     "type": "ClassDiagram",
     "nodes": [...],   ← built via make_node()
     "edges": [...],   ← built via make_edge()
     "size": {...},
     "interactive": {...},
     "assessments": {} }
```

The converter (`besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/class_diagram_converter.py:226-534`) emits v4 directly. Module docstring (lines 4-7) is explicit:

> Emits the v4 wire shape (``{nodes, edges}``) directly — see ``docs/source/migrations/uml-v4-shape.md``. There is no v3-shape intermediate; every node is built via ``make_node`` and every edge via ``make_edge``.

Hard-coded `"version": "4.0.0"` at line 526.

**Output shape: v4. No v3 anywhere in the pipeline.**

## 3. `existing_model` Round-trip

When the frontend posts an existing diagram for merge (line 646-653), the JSON is parsed via `process_class_diagram`, which (per its docstring at `class_diagram_processor.py:741-744`) **accepts only v4 `{nodes, edges}` shape and rejects v3**. So the merge path is also v4-end-to-end.

## 4. Frontend Consumer

`besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/features/import/useImportDiagramPicture.ts`

- Line 38: `fetch(${BACKEND_URL}/get-json-model-from-image, …)`
- Line 50-53: validates `data.model.type` (ClassDiagram).
- Line 30-36: when an active ClassDiagram has nodes, sends it as `existing_model` (v4 shape — `model.nodes` confirmed at line 30).
- Line 74-80: stores the returned `data.model` verbatim into the project under `diagrams[diagramType]`. No reshape, no v3 fallback.

The consumer is v4-native and matches what the backend returns.

## 5. OpenAI Prompt

The prompt (`image_to_buml.py:79-103`) requests **PlantUML class-diagram text**, not JSON in either v3 or v4 shape. PlantUML is then parsed into a DomainModel by the existing `structuralPlantUML` ANTLR parser. This is the right architectural choice — PlantUML is the LLM-stable surface; v4 JSON is generated deterministically downstream by the converter, so the prompt is **decoupled from the wire-shape version** and needs no change for v3→v4.

## 6. Issues Found

| Severity | Issue |
|----------|-------|
| _none_ | No v3↔v4 conversion at the endpoint boundary. |
| _none_ | No prompt change required (prompt asks for PlantUML, not JSON). |
| _none_ | Frontend consumer accepts the v4 response as-is. |

The user's constraint ("rejected v3↔v4 conversion in production") is not violated — there is no such conversion on this path. The pipeline emits v4 natively from the metamodel.

## 7. Recommended Fix

**None.** The endpoint is already v4-correct. No diff produced.

## 8. Cross-References

- v4 shape spec: `docs/source/migrations/uml-v4-shape.md`
- Twin endpoint (knowledge graphs): `/get-json-model-from-kg` at `conversion_router.py:689-727` shares the same `class_buml_to_json` tail and is therefore also v4-correct.
- Twin tail used by all DomainModel-producing endpoints (BUML import, CSV reverse, image, KG): single converter `class_buml_to_json` — v4 emission is centralised, low surface for regression.
