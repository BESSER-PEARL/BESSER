# GUI Metamodel Round-trip Checklist

This note tracks GUI features that historically relied on `_frontend_meta` instead of the BUML metamodel, and documents which pieces of the pipeline (JSON -> BUML -> generated code) now preserve their data end-to-end.

## Completed

- **Images**  
  Embedded image sources survive round-trip via the new `Image.source` attribute, and both the React generator plus the Python code builder emit the URI/base64 data.

- **Links**  
  Anchors now materialise as `Link` components with label, URL, target, and rel information carried through the parser, generator, and code builder layers.

- **Embedded Content**  
  Maps/iframes instantiate `EmbeddedContent`, storing the embed source (and optional content type) so the generated UI renders without consulting the original JSON.

- **Menus**  
  `MenuItem` instances retain link metadata (`url`, `target`, `rel`). The extractor, React JSON, and generated Python all include these fields.

- **Data Lists**  
  `DataSourceElement` keeps the backing class, selected fields, and label/value bindings (with string fallbacks when the domain model is absent). React payloads and generated code now expose that information.

## Still Pending

- **Generic Components (Fallback)**  
  Any widget that still falls back to a bare `ViewComponent` keeps its semantics in `_frontend_meta`. For each remaining component we adopt the same playbook—extend the metamodel, add a parser, and teach the serializers/code builder about it.

## Testing & Tooling

- Add parser unit tests covering minimal GrapesJS fragments so we can assert the produced BUML objects contain the expected data.
- Extend generator regression tests to compare regenerated JSON with the source (allowing stylistic diffs).
- Consider a CI guard that fails when `_frontend_meta` retains more than debugging artefacts (e.g., DOM class lists) after serialisation.

With these items in place, the GUI modelling pipeline is free from hidden metadata dependencies and fully round-trippable.
