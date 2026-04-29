# Issue #517 — Abstract / Enumeration toggle not clearing class annotation

**Issue:** [BESSER-PEARL/BESSER#517](https://github.com/BESSER-PEARL/BESSER/issues/517)
**Frontend PR:** [BESSER-PEARL/BESSER-Web-Modeling-Editor#116](https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor/pull/116) (merged into `develop`)

## Bug

In the BESSER Web Modeling Editor's class-diagram property panel, enabling the *Abstract* (or *Enumeration*) switch correctly added the `<<abstract>>` / `<<enumeration>>` stereotype banner above the class name. Disabling the switch, however, did **not** remove the banner — it only disappeared after a full page refresh.

## Root cause

The bug lived in `ClassifierUpdate.toggle` at:

`packages/editor/src/main/packages/common/uml-classifier/uml-classifier-update.tsx`

Two interacting problems:

1. **Stale `type` argument.** The new classifier instance was constructed with `type: element.type` (the *old* type). `UMLClassifier`'s constructor calls `assign(this, values)`, which overwrote the new class's `type` field initializer. The result: a freshly-built `UMLClass` instance ended up carrying `type === AbstractClass`.

2. **Incomplete serialization.** `UMLClassifier.serialize()` does not include `stereotype`, `italic`, or `underline`. The toggle dispatched `update(id, instance.serialize())`, so those distinguishing fields were never present in the payload. The reducer performs a partial merge, so the previous `stereotype: 'abstract'` and `italic: true` simply remained on the existing element — the visible banner persisted.

## Fix

In `toggle`:

- Removed the stale `type: element.type` argument from the constructor call so the new class's field initializer wins.
- Explicitly forwarded `stereotype`, `italic`, and `underline` from the new instance into the dispatched `update()` payload so the partial merge fully overwrites the classifier kind.

```ts
private toggle = (type: keyof typeof ClassElementType) => {
  const { element, update } = this.props;
  const newType: UMLElementType = element.type === type ? ClassElementType.Class : type;
  const instance = new UMLElements[newType]({
    id: element.id,
    name: element.name,
    owner: element.owner,
    bounds: element.bounds,
    ownedElements: element.ownedElements,
  }) as UMLClassifier;
  const { id: _ignoredId, ...values } = instance.serialize();
  update<UMLClassifier>(element.id, {
    ...values,
    stereotype: instance.stereotype,
    italic: instance.italic,
    underline: instance.underline,
  } as Partial<UMLClassifier>);
};
```

## Verification

- Drop a class → enable Abstract → `<<abstract>>` appears.
- Disable Abstract → stereotype is removed immediately, no refresh required.
- Same behavior verified for the Enumeration toggle.
- Toggling Abstract ↔ Enumeration swaps stereotypes correctly.
- `tsc --noEmit` reports no new errors on the touched file.

## Follow-up

The fix lives in the frontend submodule. After the next release cut, the BESSER backend's submodule pointer at `besser/utilities/web_modeling_editor/frontend` should be advanced to a `main` commit that contains this change (it currently rides on `develop`).
