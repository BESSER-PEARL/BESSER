# Third-Party Licenses

BESSER is distributed under the MIT License (see `LICENSE.md`). This file
lists third-party software that BESSER redistributes or bundles as part of
its build/runtime artifacts, along with its license.

## Alloy Analyzer

- **Component**: [Alloy Analyzer](https://alloytools.org/) 6.2
  (`org.alloytools.alloy.dist.jar`)
- **Used by**: `besser/generators/alloy/` (Alloy specification generator) and
  its object-diagram / consistency-check backend
  (`besser/generators/alloy/instance_generator/`)
- **License**: MIT License
- **Copyright**: Copyright (c) the Alloy project and contributors
- **License text**: <https://github.com/AlloyTools/org.alloytools.alloy/blob/v6.2.0/LICENSE>
- **Source / release**: <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>

Alloy Analyzer is downloaded and bundled into the `besser` Docker image at
build time (see `Dockerfile`), where its jar is placed at `/alloy/` alongside
a copy of its MIT license text (`/alloy/LICENSE`), fulfilling the license's
notice requirement for that redistributed copy.

When running BESSER outside Docker, the jar is not vendored in this
repository; it is installed and configured separately by the user (via the
`JAVA_HOME` and `BESSER_ALLOY_JAR` environment variables, see
`docs/source/generators/object_diagram.rst`) and invoked by BESSER as an
external process at runtime.
