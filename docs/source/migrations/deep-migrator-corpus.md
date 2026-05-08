# SA-DEEP-MIGRATOR-CORPUS sweep

Real production v3 fixtures piped through the per-diagram-type v3→v4 migrators.
Counts only data fields that **were populated in v3** but vanished in v4.

## Totals

- Fixtures discovered: **38**
- PASS: **35**
- WARN (data loss): **2**
- FAIL (crash / count drop): **0**
- SKIP (no migrator for type): **1**
- Nodes preserved: **248** / **248**
- Edges preserved: **219** / **219**

## By diagram type

| Type | PASS | WARN | FAIL | SKIP |
|------|-----:|-----:|-----:|-----:|
| AgentDiagram | 13 | 0 | 0 | 0 |
| ClassDiagram | 14 | 0 | 0 | 0 |
| CommunicationDiagram | 0 | 0 | 0 | 1 |
| NNDiagram | 4 | 0 | 0 | 0 |
| ObjectDiagram | 1 | 0 | 0 | 0 |
| StateMachineDiagram | 2 | 0 | 0 | 0 |
| UserDiagram | 1 | 2 | 0 | 0 |

## Top data-loss patterns

| # | Count | Pattern |
|--:|------:|---------|
| 1 | 9 | `UserDiagram drops attribute.visibility (by-design? v3=…)` |

## Per-fixture results

| Status | Type | Nodes | Edges | Issues | Fixture |
|--------|------|------:|------:|------:|---------|
| PASS | AgentDiagram | 3/3 | 3/3 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/agent/dbagent.json` |
| PASS | AgentDiagram | 5/5 | 4/4 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/agent/faqragagent.json` |
| PASS | AgentDiagram | 17/8 | 6/6 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/agent/greetingagent.json` |
| PASS | AgentDiagram | 11/9 | 8/8 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/agent/gymagent.json` |
| PASS | AgentDiagram | 20/10 | 8/8 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/agent/libraryagent.json` |
| PASS | NNDiagram | 20/3 | 14/14 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/nn/alexnet_nn.json` |
| PASS | NNDiagram | 7/1 | 5/5 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/nn/lstm_nn.json` |
| PASS | NNDiagram | 12/4 | 10/10 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/nn/tutorial_example.json` |
| PASS | ClassDiagram | 5/5 | 2/2 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/project/library_full_stack.json::diagrams.ClassDiagram[0]` |
| PASS | AgentDiagram | 20/10 | 8/8 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/project/library_full_stack.json::diagrams.AgentDiagram[0]` |
| PASS | AgentDiagram | 11/9 | 9/9 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/project/personalized_gym_agent.json::diagrams.AgentDiagram[0]` |
| WARN | UserDiagram | 2/2 | 1/1 | 6 | `<develop>/packages/webapp/src/main/templates/pattern/project/personalized_gym_agent.json::diagrams.UserDiagram[0]` |
| WARN | UserDiagram | 3/3 | 2/2 | 3 | `<develop>/packages/webapp/src/main/templates/pattern/project/personalized_gym_agent.json::diagrams.UserDiagram[1]` |
| PASS | StateMachineDiagram | 13/7 | 4/4 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/statemachine/traficlight.json` |
| PASS | ClassDiagram | 3/3 | 2/2 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/Library.json` |
| PASS | ClassDiagram | 5/5 | 2/2 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/Library_Complete.json` |
| PASS | ClassDiagram | 15/15 | 13/13 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/Library_OCL.json` |
| PASS | ClassDiagram | 26/26 | 31/31 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/ai_sandbox.json` |
| PASS | ClassDiagram | 8/8 | 7/7 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/dpp.json` |
| PASS | ClassDiagram | 18/18 | 19/19 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/nexacrm.json` |
| PASS | ClassDiagram | 4/4 | 3/3 | 0 | `<develop>/packages/webapp/src/main/templates/pattern/structural/team_player_ocl.json` |
| PASS | ClassDiagram | 3/2 | 1/1 | 0 | `<develop>/packages/editor/src/tests/unit/test-resources/class-diagram-2.json` |
| PASS | ClassDiagram | 2/2 | 1/1 | 0 | `<develop>/packages/editor/src/tests/unit/test-resources/class-diagram-3.json` |
| PASS | ClassDiagram | 3/3 | 1/1 | 0 | `<develop>/packages/editor/src/tests/unit/test-resources/class-diagram-4.json` |
| PASS | ClassDiagram | 3/2 | 1/1 | 0 | `<develop>/packages/editor/src/tests/unit/test-resources/class-diagram.json` |
| SKIP | CommunicationDiagram | — | — | 0 | `<develop>/packages/editor/src/tests/unit/test-resources/communication-diagram.json` |
| PASS | ClassDiagram | 42/42 | 31/31 | 0 | `<develop>/packages/editor/src/main/packages/user-modeling/usermetamodel_buml_less_short.json` |
| PASS | AgentDiagram | 11/6 | 5/5 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentDiagram.json` |
| PASS | AgentDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentTransitionShape1.json` |
| PASS | AgentDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentTransitionShape2.json` |
| PASS | AgentDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentTransitionShape3.json` |
| PASS | AgentDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentTransitionShape4.json` |
| PASS | AgentDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/agentTransitionShape5.json` |
| PASS | ClassDiagram | 4/4 | 2/2 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/classDiagram.json` |
| PASS | NNDiagram | 8/5 | 5/5 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/nnDiagram.json` |
| PASS | ObjectDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/objectDiagram.json` |
| PASS | StateMachineDiagram | 13/10 | 4/4 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/stateMachineDiagram.json` |
| PASS | UserDiagram | 2/2 | 1/1 | 0 | `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3/userDiagram.json` |

## Recommended migrator fixes

Order matches the top-pattern table above:

1. **UserDiagram drops attribute.visibility (by-design? v3=…)** _(seen 9×)_ — By design — UserDiagram v4 does not model visibility (its attributes are operator comparisons). Confirm with product before treating as a bug; otherwise add `visibility` to the v4 row schema.

## Source roots scanned

- `<develop>/packages/library/tests/fixtures/v3` (missing — skipped)
- `<develop>/packages/webapp/src/main/templates` (present)
- `<develop>/packages/editor/src/tests/unit/test-resources` (present)
- `<develop>/packages/editor/src/main/packages/user-modeling` (present)
- `<repo>/besser/utilities/web_modeling_editor/frontend/packages/library/tests/fixtures/v3` (present)
- `<repo>/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/templates` (present)
- `<repo>/besser/utilities/web_modeling_editor/frontend/packages/editor/src/tests/unit/test-resources` (present)
- `<repo>/besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/user-modeling` (present)
