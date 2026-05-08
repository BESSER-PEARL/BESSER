# PC-4: Object node + inspector

Read-only audit of the ObjectDiagram visual node and its inspector.

## Sources

### Old (`packages/editor/src/main/packages/uml-object-diagram/`)
- `uml-object-name/uml-object-name.ts` (model — `classId`, `className`, `icon`, child reordering, dual-view render)
- `uml-object-name/uml-object-name-component.tsx` (SVG renderer — header band, underline, optional stereotype, icon view branch)
- `uml-object-name/uml-object-name-update.tsx` (popup — class dropdown via `diagramBridge.getAvailableClasses()`, auto-attribute population, name placeholder logic)
- `uml-object-attribute/uml-object-attribute.ts` (`attributeId`, `attributeType`, `displayName` strips visibility/type)
- `uml-object-attribute/uml-object-attribute-update.tsx` (split view, type-aware widgets: enum dropdown, date/datetime/time inputs, duration free-form, str quote-wrap)
- `uml-object-method/uml-object-method.ts` (thin extension of `UMLClassifierMethod`)
- `uml-object-icon/uml-object-icon.ts` (separate child element holding the SVG body, layout-only)

### New (`packages/library/lib/`)
- `nodes/objectDiagram/ObjectName.tsx` (React-Flow node wrapper, layout, `formatObjectAttribute` derives `name = value` for canvas)
- `components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx` (header `isUnderlined={true}`, `iconViewActive` via `useSettingsStore`, `foreignObject` for inline SVG body)
- `components/inspectors/objectDiagram/ObjectEditPanel.tsx` (class picker + per-row type-aware widgets + auto-populate)
- `types/nodes/NodeProps.ts` — `ObjectNodeProps { methods, attributes, classId?, className?, icon? }`, `ObjectNodeAttribute = ClassNodeElement & { attributeId?, value? }`
- `utils/classifierMemberDisplay.ts` (`formatDisplayName` — UML mode emits visibility symbol)

## Verdict

**Substantial parity, with three behavioural gaps.** The new lib reproduces the load-bearing behaviours: classId picker via `diagramBridge.getAvailableClasses()`, auto-attribute population on class change (`handleClassChange`, lines 423–471 mirroring v3 lines 107–128), inline icon-view via `useSettingsStore` + `dangerouslySetInnerHTML`, header underline, and the SA-2.1 type-aware value widgets (enum / date / datetime / time / duration / quoted str / plain) — see `ObjectEditPanel.tsx:130–225`. Round-trip-relevant fields (`classId`, `className`, `icon`, per-row `attributeId`, `attributeType`, `value`) are all retained on `ObjectNodeProps` / `ObjectNodeAttribute`.

## Coverage matrix

| Feature                                                                 | Old                              | New                                   | Notes                                                                                                          |
| ----------------------------------------------------------------------- | -------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Underline on object name                                                | hard-coded `textDecoration` (component lines 110, 126) | `HeaderSection isUnderlined={true}` (`ObjectNameSVG.tsx:84`) | Parity. |
| Stereotype band (`«…»` line above name)                                 | renders when `element.stereotype` (component lines 104–120) | hard-coded `showStereotype={false}, stereotype={undefined}` (`ObjectNameSVG.tsx:78–79`) | **Gap 1** — new lib drops stereotype on object names. |
| Icon view when `data.icon` set                                          | settings toggle + child icon's `icon` content (`uml-object-name.ts:146–204`) | `useSettingsStore(s => s.showIconView)` + `data.icon` (`ObjectNameSVG.tsx:40–42, 92–114`) | Parity. New lib stores SVG body on the node itself instead of a separate `ObjectIcon` child. |
| Header band                                                             | two `ThemedRect`s + divider path | `HeaderSection` + outer `StyledRect` | Parity. |
| Attributes / methods as rows on `data.attributes` / `data.methods`      | `UMLObjectAttribute` / `UMLObjectMethod` children, layout via `UMLClassifier.render` | `RowBlockSection` per array (`ObjectNameSVG.tsx:117–155`) | Parity (data shape: `ObjectNodeProps.attributes/methods`). |
| Display row as `attribute = value` on canvas                            | popup writes `name = value` then row.displayName returns raw name | `formatObjectAttribute` derives label from `value` if not already inlined (`ObjectName.tsx:24–31`) | Parity, new lib is more forgiving. |
| classId picker dropdown driven by `diagramBridge.getAvailableClasses()` | `getAvailableClasses` with hierarchy / attr-count formatter (popup lines 59–79) | MUI `Select` with `(N attrs)` suffix (`ObjectEditPanel.tsx:584–604`) | Mostly parity. **Gap 2** — `getClassDisplayName` in v3 also showed `extends Parent` via `getClassHierarchy(...)`; new dropdown shows attr count only. |
| Stereotype field on the inspector                                       | not surfaced in v3 popup either   | not surfaced                          | n/a — v3 also omits. |
| Name field                                                              | `Textfield` w/ `getDisplayName` placeholder | `NodeStyleEditor` w/ `placeholderName` (`ObjectEditPanel.tsx:569–580`) | Parity. |
| Auto-attribute population on classId pick                               | `onClassChange` deletes existing then creates from `selectedClass.attributes` (lines 80–129) | `handleClassChange` replaces `attributes` with new rows, sets `classId` / `className`, renames if name was empty / "Object" (`ObjectEditPanel.tsx:423–471`) | **Confirmed shipped.** Includes inherited (`getAvailableClasses` flattens chain). |
| Per-row type-aware widget — enum                                        | `isEnumerationAttribute` + Dropdown (lines 117–142, 332–344) | `matchingEnum` lookup + MUI `Select` (`ObjectEditPanel.tsx:121–124, 132–152`) | **Confirmed shipped (SA-2.1 #5).** |
| Per-row type-aware widget — date / datetime / time                      | `DateTimeInput` styled native input, format helpers (lines 145–176, 240–296, 350–357) | MUI `TextField type={dateKind}` (`ObjectEditPanel.tsx:156–171`) | **Confirmed shipped.** Same kind set (date/localdate, datetime/timestamp/localdatetime/offsetdatetime/zoneddatetime/instant, time/localtime/offsettime). |
| Per-row type-aware widget — timedelta / duration free-form              | `renderDurationInput` w/ placeholder `"e.g., 1d 2h 30m, P1DT2H30M, 1:30:00"` | identical placeholder in MUI TextField (`ObjectEditPanel.tsx:174–189`) | **Confirmed shipped.** |
| Per-row type-aware widget — `str` quote-wrap                            | `QuoteWrapper` with `<Quote>"</Quote>` siblings, raw value stored unquoted (lines 376–388) | display `"${value}"`, strip surrounding quotes on edit (`ObjectEditPanel.tsx:193–208`) | **Confirmed shipped**, semantically equivalent. |
| Per-row type-aware widget — fallback plain text                         | `ValueTextfield` (line 366–374)  | MUI `TextField` (`ObjectEditPanel.tsx:211–224`) | Parity. |
| Visibility shows symbols only on rows                                   | `UMLObjectAttribute.displayName` returns bare `name` — visibility never rendered | object rows route through `RowBlockSection` which treats them as `ClassNodeElement` and `formatDisplayName` prepends a visibility symbol when `mode==='UML'` | **Gap 3** — see below. |
| Linked attributeId selector per row                                     | not surfaced in v3 popup (only via library auto-populate) | explicit `Select` with `linkedClassAttrs` (`ObjectEditPanel.tsx:304–324`) | New lib improvement. |
| `attributeType` chooser per row                                         | not surfaced in v3 popup          | full primitive list + classes + enumerations + custom (`ObjectEditPanel.tsx:248–280`) | New lib improvement. |
| Methods per-row editor                                                  | section commented out in v3 popup | full add/edit/delete UI (`ObjectEditPanel.tsx:642–680`) | New lib improvement. |
| Style pane (fill / line / text colour)                                  | `StylePane` (per-element + per-attribute) | `NodeStyleEditor` for the node | Parity at the node level; per-row colour pane dropped. |
| Trash / delete attribute                                                | per-row trash button             | `IconButton DeleteIcon` (`ObjectEditPanel.tsx:281–285`) | Parity. |

## Top 3 gaps

1. **Object-name stereotype band is hard-coded off.** `ObjectNameSVG.tsx:78–79` passes `showStereotype={false}` and `stereotype={undefined}`. v3 read `element.stereotype` and rendered a `«stereotype»` line above the underlined name (component lines 104–120) plus widened the header to 50px. Any v3 model that set a stereotype on an object instance loses that label after migration. If stereotypes on object instances are intentionally unsupported in v4, document the cut; otherwise the new SVG and `ObjectNodeProps` need a `stereotype` field plumbed through.
2. **classId dropdown lost the inheritance hint.** v3 `getClassDisplayName` rendered `Foo extends Bar, Baz (3 attrs)` using `diagramBridge.getClassHierarchy(cls.id)`; new lib shows only `Foo (3 attrs)` (`ObjectEditPanel.tsx:597–602`). Acceptable but a UX regression when picking among similarly-named subclasses.
3. **Visibility symbols leak onto object-instance rows.** v3 explicitly overrode `UMLObjectAttribute.displayName` to strip visibility/type (`uml-object-attribute.ts:23–25`), since UML object diagrams render `attribute = value` only. The new lib reuses `RowBlockSection` + `formatDisplayName` — which prepends `+`/`-`/`#`/`~` whenever `mode==='UML'` and the row has both `name` + `attributeType`. With the new inspector setting `attributeType` (and visibility defaulting to `"public"` on `addAttribute`, line 512), the canvas will display e.g. `+ counter: int = 5` instead of `counter = 5`. The label-derivation in `formatObjectAttribute` (`ObjectName.tsx:24–31`) only short-circuits when `name` already contains `" = "` — but rows entered through the inspector store `name="counter"` and `value="5"` separately, so they hit `formatDisplayName` via `processedAttributes`. Either route object rows through a Chen-/object-style formatter (no visibility, `name = value`) or override `formatObjectAttribute` to fully short-circuit `formatDisplayName`.

## Files

- Old: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-object-diagram/{uml-object-name,uml-object-attribute,uml-object-method,uml-object-icon}/`
- New: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/objectDiagram/ObjectName.tsx`, `library/lib/components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx`, `library/lib/components/inspectors/objectDiagram/ObjectEditPanel.tsx`, `library/lib/types/nodes/NodeProps.ts`, `library/lib/utils/classifierMemberDisplay.ts`
