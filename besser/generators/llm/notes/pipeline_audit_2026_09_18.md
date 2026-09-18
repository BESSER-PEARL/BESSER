# Spec-driven pipeline audit — 18 September 2026

The two-stage approach is sound. The largest remaining opportunity is to make
the modeling agent use BESSER's existing capabilities, preserve every requirement
between stages, and verify the resulting application through executable user
workflows. Additional repair prompts alone cannot supply those guarantees.

This is a local code review and reproduction report, not a completed hotel app
or a deployment certification. The hotel prompt is the acceptance case for the
pipeline. Repositories inspected: BESSER at `31b02f26`, modeling-agent at
`e715ce7`, and the editor submodule at `f5749b38`, including working changes.
Another session was actively editing the working trees during inspection.
The ledger, action-route changes, and new modeling fidelity changes were
uncommitted during the audit; findings concerning them describe work in progress.

Verification performed:

- 122 focused modeling-agent tests passed: multiplicities, association merging,
  mandatory cycles, and the hotel model contract.
- 48 focused BESSER tests passed: ledger, OCL task seeding, constructibility,
  instance action routes, atomic creation, and method-button binding.
- 24 existing generated-backend tests passed: native association classes and
  relationship lower bounds, including actual HTTP requests against SQLite.
- `verification/spec_driven_audit_2026_09_18.py`, relative to the workspace root,
  reproduced the additional findings below. It uses temporary applications,
  makes no LLM calls, and does not modify repository application code.

These 194 passing tests verify the existing checks' behavior. Several checks
still encode an inadequate definition of success. No fresh live model run,
full generated React build, browser acceptance campaign, or deployment was
performed in this audit.

**What today's changes improve**

Keep the per-instance multiplicity fields, reciprocal-association cleanup,
OCL transport through the editor, atomic create changes, preservation of the
original request, derived-attribute serialization, actionable edit diagnostics,
mapper checks, and repair-loop improvements. The new instance-action routes
address a real mismatch with the generated React buttons. The requirements
ledger is a useful audit mechanism, subject to the limitations below.

**1. Use native association classes before adding aggregate workarounds.**

BESSER already supports an attributed relationship and creating its link rows
inside the parent's request. In
`tests/generators/backend/test_backend_assoc_class.py:162`, a generated Trip
accepts two seats with different agreed prices in one POST. The hotel-shaped
`test_backend_lower_bounds.py` additionally verifies that the last required
guest or room cannot be removed through several mutation paths.

The modeling agent's complete-system schema exposes ordinary classes, enums,
and ordinary relationships, but not the native association-class attachment.
Its rule 17 in `src/diagram_handlers/types/class_diagram_handler.py` tells the
model to create an ordinary junction class between Booking and Room. That
misses the generator's existing attributed-link path and can create a cyclic
construction dependency.

Implement explicit association-class representation in the compact and
canonical schemas and its conversion into the editor's association-class link.
For this hotel, Booking–Room should carry the agreed price and extra charges
through that supported construct. Verify it by passing an agent spec through
the real editor converter, BUML converter, generator, and HTTP create request.
Do not automatically convert every two-parent class: some junction entities
have independent identity and lifecycle and need to remain ordinary classes.

This corrects an important diagnosis: per-entity CRUD is not inherently unable
to create this hotel aggregate. The supported model representation matters.

**2. Keep domain requirements separate from construction limitations.**

`ClassDiagramHandler._break_mandatory_cycles` changes required ends to optional
and appends an OCL invariant. `pydantic_classes/ocl_utils.py` explicitly skips
collection and relationship expressions. The invariant preserves the intent
as text, but it does not preserve runtime enforcement by itself.

The new `_unenforced_rule_tasks` in `llm/orchestrator.py` helps retain that work,
but directs checks to the context class's create/update endpoints. An aggregate
may not yet have its children there. The existing dependent-row guidance in
`gap_analyzer.py` recognizes this problem; the new deterministic seeds need the
same dependency information. Rules also need coverage when child rows or
relationships are changed or deleted.

Retain the original cardinalities in the semantic model. Record any lowering
needed by a target's persistence API separately, with an explicit enforcement
obligation. First use the existing native association-class path. Where it
cannot express the desired operation, create a transaction-level command that
builds the complete aggregate, validates it, and commits once. A draft workflow
is an application policy, not an automatic substitute for “every booking has
at least one room.”

**3. Make computed-field ownership deterministic.**

Today's serializer fix makes `is_derived` visible to the coding agent. It does
not change the generated input contract. The audit generated a class with
`totalPrice: float, is_derived=True` and confirmed:

ok yes the goal is not to chaangge besser is really to be able to ggive aall the spec to the aggent as we dont have the derived into besser we provide to the coding aghent this so it should do somethingg

```text
BookingCreate.totalPrice.is_required() == True
BookingCreate(number=1, totalPrice=999).totalPrice == 999.0
```

The server-owned filter in
`pydantic_classes/templates/pydantic_classes_template.py.j2:87` handles
surrogate IDs and audit timestamps, not derived attributes.

Generate separate write/read contracts and a clear implementation boundary for
derived values. Prevent writes to computed fields in create, update, bulk, and
relationship operations. Generate a pending implementation obligation when the
formula is unavailable; do not substitute a constant zero or silently remove a
required database value. For the hotel, business state, stay state, and total
belong to server behavior. Their exact derivation belongs to the application.

Natural identifiers need the same complete round trip. During this review the
modeling-agent working tree gained `isExternalId` and compact `!` notation.
The editor initially dropped the new flag; the other active session added
converter/modifier support before this audit finished. BESSER's SQLAlchemy
template already understands `is_external_id` and emits uniqueness. That
plumbing is now work in progress across the repositories, not an untouched
backlog item. Verify a real duplicate insertion after the entire round trip
before counting it as an end-to-end fix; this audit did not test those late edits.

**4. Separate “failed,” “passed,” and “not verified.”**

The reproduction script confirmed the following ledger behavior:

| Situation | Observed result | Required change |
|---|---|---|
| Requirement extraction fails | Empty issue list; cached as `[]`; not retried on the next collection | Preserve failure state; record incomplete verification and allow a bounded retry |
| Required behavior is partial | Warning, outside the automatic blocker fix loop | Keep the requirement incomplete until its acceptance criteria pass |
| Claimed implementation has no matching evidence | Warning | Distinguish unverified from verified; do not certify completion |
| Citation points only to `# enforce_capacity` | Remains `implemented` | Treat citations as navigation evidence, not proof of behavior |
| Behavior lives in a shared React component | Component absent from the code digest | Resolve relevant imports/files, record truncation, and run the UI test |

`requirements_ledger.py` checks whether a path exists and a symbol string occurs
in its text. That is useful for rejecting nonexistent citations but cannot
prove that a rule is enforced. Its digest reads Python and TSX/JSX under
`pages/`, omitting shared components, hooks, and TypeScript API clients. Raising
the character cap cannot correct those omissions. The 40-item requirement cap
also needs an explicit overflow result rather than silently discarding scope.

Extract the requirement list before model construction/customization. Preserve
stable IDs, source quotations, model mapping, implementation owner, verification
method, and evidence revision. Keep the stack requirements and ordinary data
requirements in that contract too; the ledger currently asks the extractor to
skip them. Use the LLM judge to find likely omissions and navigate code. Use
tests and declared generator guarantees for the completion decision.

Another reproduced defect concerns stage persistence. `_collect_model_contract_issues`
records a mandatory-cycle blocker, but `_run_phase3_validation` later assigns
`self._validation_issues = list(issues)`. With an unrelated style finding in
Phase 3, the model blocker disappeared. Keep findings keyed by stage/check and
replace only the results of the check rerun. The model check currently records
a blocker and continues generation; it is not an immediate generation stop.

Validation also returns early if the runtime budget is already exhausted.
Reserve time for final checks and represent skipped checks explicitly. A skipped
check need not allege that the code is defective, but must prevent a claim that
all requirements were verified.

**5. Fix the creation probe's inference before expanding its repair authority.**

The audit generated a working Room API, then added the legitimate business rule
`roomNumber >= 100`. The constructibility probe reported:

```text
POST /room/ cannot create a Room — every schema-valid request is rejected
sample payload: {"roomNumber": 1, "status": "AVAILABLE"}
```

An actual request with room number `101` returned HTTP 200. `_sample` always
uses `1` for numbers; `_variants` changes enum/boolean/date values, not numeric
business-rule inputs. Rejection of those samples cannot prove impossibility.
Worse, the resulting repair advice suggested moving the valid create-time rule
because Booking happens to depend on Room.

Also reproduced: all-422 and all-unresolved reports produce no findings, and
`_extract_id({"roomNumber": 101})` returns `None`. The probe knows only `id`,
although BESSER supports custom primary keys.

Retain the probe as a smoke check. Generate known-valid examples from the
requirement/model contract, respect schema bounds, support real ORM primary
keys and association-class payloads, and report attempts/coverage. A sampled
400 is inconclusive unless the harness knows the input satisfies the business
preconditions. Do not instruct the agent to relax a requirement to satisfy a
smoke test. Failed HTTP 500s, demonstrated contract violations, and failed
known-valid workflows warrant different diagnostics from unknown test data.

**6. Run generated code in an isolated execution worker.**

`orchestrator._import_smoke_issues` imports generated Python with the host
interpreter; `constructibility._run_probe` executes a temporary copy with that
interpreter. Both use a filtered environment and a timeout. Neither call sets
up a separate filesystem/network security boundary. This is visible in the
code; this audit did not attempt a sandbox escape or inspect the live host.

The capability to boot and test an app is necessary. Put it in a per-run worker
with only the generated workspace and a disposable database, no control-plane
credentials, resource limits, and controlled network access. Keep the hosted
arbitrary-shell default off. The agent can receive narrow build/test/browser
operations backed by that worker. Install the application's declared
dependencies there so missing host dependencies are not confused with app
quality. gVisor is an existing OCI-compatible option for strengthening workload
isolation; adopting it needs testing against this deployment, not a custom
sandbox implementation. [gVisor architecture](https://gvisor.dev/docs/architecture_guide/intro/)

**Where each responsibility belongs**

| Owner | Reusable responsibilities | Hotel example |
|---|---|---|
| Modeling agent and converters | Faithful concepts, inheritance, both relationship directions and roles, identifiers, association classes, derived markers, constraints, requirement references | Contact is a Person; staying guests are a separate relationship; price belongs to Booking–Room |
| Deterministic BESSER generators | ORM/schema/route contracts, supported validation, database uniqueness, atomic persistence, relationship bounds, typed action endpoints, stable extension points | A booking accepts guest IDs and room-price link payloads; room numbers are unique |
| Spec-driven orchestration | Capability manifest, unresolved requirement tasks, bounded edits, actual execution feedback, acceptance evidence, honest completion and resume | Capacity and overlap checks remain open until tested |
| Generated application extensions | Domain policies, calculations, transaction-level operations, user workflow | Billing, payment, cancellation, check-in/out, extra charges, availability and totals |
| External tooling | Parsing/linting, API schema testing, browser execution, property-based testing, workload isolation | Detect route failures, invalid inputs, UI dead ends, and bad action sequences |

Generate a machine-readable capability/obligation manifest during Phase 1.
For each requirement distinguish `enforced`, `scaffolded`, and `unsupported`,
with implementation location and the check that establishes that status. A
method returning 501 is scaffolded, not implemented. A regex compiled into a
validator has a different guarantee from a relationship rule left as a comment.
The gap plan should be the remaining obligations plus application-specific
behavior, not a fresh guess about which code the templates emitted.

Define stable extension modules for services/actions/validators/derived-value
providers. CRUD, bulk, relationship, and action routes should call shared policy
hooks. The agent should not have to copy a capacity guard into numerous router
functions. Preserve generated ownership and extension code across regeneration.
Keep a way to make a necessary model change and regenerate dependent artifacts;
an omitted field should not force the agent to contradict a frozen wrong model.

The hotel examples in the modeling prompt are useful regression cases, but
should not keep expanding into the general runtime instructions. In particular,
the example capacity formula refers to `guestCount` and direct `rooms`, while
the user's requirement is about the listed guests and the actual room links.
The overlap example does not mention the active-booking predicate. Resolve
expressions against actual model roles, preserve the source rule, and validate
the result. Move long incident narratives into fixtures/docs. Keep concise
generic rules in prompts and structured guarantees in schemas/tools.

**External components worth reusing**

- Use pytest/httpx for known hotel scenarios. Add
  [Schemathesis](https://schemathesis.readthedocs.io/en/stable/reference/checks/)
  for OpenAPI response and linked-operation checks. Supply links and fixtures;
  schema fuzzing cannot infer the capacity, billing, or cancellation policies.
- Use [Hypothesis stateful tests](https://hypothesis.readthedocs.io/en/latest/stateful.html)
  for sequences such as bill → pay → cancel → pay again, with explicit domain
  assertions. It generates action sequences; BESSER still supplies their meaning.
- Use [Playwright](https://playwright.dev/docs/test-assertions) on the generated
  app for the actual receptionist workflow, refusal messages, keyboard access,
  and refresh/persistence. Editor UI tests do not prove generated-app usability.
- Generate a typed client from a sufficiently typed OpenAPI contract using
  [openapi-typescript/openapi-fetch](https://openapi-ts.dev/openapi-fetch/api).
  Keep endpoint definitions, instance ID types, request models, and action
  responses consistent first; pervasive `response_model=None` limits the value
  of client generation.
- Keep standard Python/TypeScript parsers and compilers for structural checks.
  The new `MethodButton` JSX regex is a useful check for known generated syntax,
  but does not replace compilation and a browser test. Treat these tools as
  focused components; replacing the entire orchestrator is not justified by
  the evidence collected here.

**Hotel acceptance contract**

The hotel should be a fixed benchmark fixture with assertions independent of
the implementation agent's own checklist. The reusable runner executes these
app-specific cases.

| Area | Required successful behavior and refusal cases |
|---|---|
| People | Person can exist without a booking; Guest and Employee reuse the same personal details and identity semantics; all three creation/update paths enforce email and phone rules |
| Identifiers | Duplicate person, room, booking and bill identifiers are rejected; references use the correct key types |
| Contacts and staff | Exactly one contact and employee per booking; contact may be absent from staying guests; either may handle multiple bookings |
| Guests | One or more guests; reuse across bookings works; duplicate submitted IDs cannot inflate occupancy counts |
| Rooms and prices | One or more distinct rooms; agreed prices differ by booking; changing Room's standard price does not overwrite existing agreements |
| Dates | Arrival before or equal to departure accepted; reversed dates refused |
| Capacity | Exact capacity accepted; exceeding it refused; edits to guests, room links and relevant capacities cannot bypass the rule |
| Availability | Conflicting active bookings refused; cancellation releases availability; edits and concurrent requests cannot introduce conflicts |
| Commercial state | Starts awaiting payment; settlement confirms; cancellation cancels; direct state assignment is refused or ignored according to an explicit input contract |
| Physical state | Starts not arrived; check-in and check-out produce the stated transitions; invalid/repeated transitions have a defined refusal result |
| Money | Amount uses agreed room prices, booked nights and room-linked extras; zero-night stays remain valid; decimal/cents arithmetic and rounding policy are consistent |
| Billing | Booking can have no bill; at most one bill; bill references exactly one booking; payment updates bill and booking atomically; repeated payment has an explicit result |
| UI and persistence | Complete workflow from an empty database with useful labels, selectors, defaults and messages; actions address the correct entity; reload and restart retain data; narrow-screen and keyboard flow work |
| Mutation safety | Failed create/update/action rolls back completely; bulk and relationship endpoints cannot bypass required invariants |

Some product decisions are not stated by the prompt: precise cancellation
eligibility, whether payment is required before check-in, bill recalculation
after issuance, and how a zero-night stay occupies a room. Record chosen
assumptions in the app specification and tests. Do not label those choices as
requirements explicitly supplied by the user. Likewise, this request does not
specify authentication, online card processing, taxes, or refunds.

For SQLite overlap protection, checking availability and then committing in
separate transactions is insufficient. Design a serialized check-and-write
transaction and handle contention. SQLite allows only one simultaneous writer;
`BEGIN IMMEDIATE` acquires the write transaction before subsequent checks but
can return `SQLITE_BUSY`. Test the chosen SQLAlchemy/SQLite integration and retry
policy with concurrent requests. [SQLite transactions](https://www.sqlite.org/lang_transaction.html)

**Implementation order**

1. Correct completion/evidence semantics, preserve findings across stages, and
   stop the creation probe from directing repairs on unproven impossibility.
   Isolate generated-code execution before expanding hosted runtime validation.
2. Complete the model/converter contract: native association classes, both roles,
   identifiers, derived ownership, and lossless constraints. Prove each with
   conversion-to-generated-HTTP tests.
3. Extract the authoritative requirement contract early and emit a Phase-1
   capability manifest. Use this to seed pending behavior and protect fulfilled
   requirements during repairs and regeneration.
4. Add common transaction/policy extension points and typed action contracts.
   Implement hotel policies in generated application extensions.
5. Require a clean dependency install/build, known-valid and invalid API cases,
   and the browser workflow before reporting the app verified. Preserve test
   artifacts and failures for the next repair iteration.
6. Evaluate repeated runs across hotels and other domains with independently
   maintained assertions. Track model fidelity, startup, requirement coverage,
   UI completion, false blockers, repair regressions, latency and cost
   separately. Keep final acceptance tests outside the coding agent's writable
   area; freeze their expected outcomes before implementation.

The next success criterion should be a complete, repeatable hotel workflow with
all required refusals verified, followed by the same pipeline succeeding on
another domain. A larger count of generated routes or passing harness unit
tests is not sufficient evidence of that outcome.
