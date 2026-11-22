# Study cleanup and development integration

This branch integrates `origin/development` at `f123a2ba` with the personalization changes.

## Removed

- The study deployment router and its website deployment tests. This removes the study-only agent and website deployment endpoints, participant model submission and storage, study agent process lifecycle, and study-specific environment settings used by the router.
- The backend startup/shutdown hooks and router registration for the study deployment, and a hard-coded developer-specific import path.
- No PostHog integration was added by this backend branch; PostHog collection was handled by the frontend and is documented in that repository's `STUDY_CLEANUP.md`.

## Adapted to development

- Kept development's refactored generation dispatch, traced output, authentication, error handling and agent template; integrated the branch's GUI personalization, text-to-speech, user profile, conversational tone and explanation rules into those paths.
- Preserved both non-equality criteria and multiple-user profile JSON export. Multiple profiles are returned as a ZIP of named JSON files; a single profile remains a single-file download.
- Combined GUI personalization styles with development's stylesheet preservation, retaining per-element rules and the CSS cascade.
- Kept development's GitHub session requirement for agent recommendations; the companion frontend branch now sends the session through the shared API client.
- Kept development's dependency changes, using its root requirements and the backend's updated dependencies. The branch retains its OpenAI SDK range needed by its text-to-speech code.
- Added tests for multi-profile export and criteria, and retained both branches' agent and recommendation tests.

Development already contains a separate opt-in research telemetry API. It is not the study-deployment feature removed here and remains as part of development.

- The backend's embedded frontend submodule follows development's pointer. The separate frontend feature branch contains the matching personalization UI and must be reviewed and merged separately.
