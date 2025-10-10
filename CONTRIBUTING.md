# Contributing to this Repository

First of all — **thank you** for your interest in contributing to our project! ❤️  
There are several ways you can help, beyond writing code.  
This document provides a high-level overview of how to get involved and how your contributions will be handled.

---

## 🧠 Asking Questions

Have a question? Feel free to [open an issue](https://github.com/BESSER-PEARL/BESSER/issues/new/choose).

Project maintainers will be happy to help.  
Please write **clear and concise questions** — the more detail you provide, the better we can assist you.

---

## 🐞 Reporting Issues or Requesting Features

Found a bug? Have a feature request? We want to hear from you!  
Please follow these steps to help us handle your report effectively:

### 1. Check for Existing Issues
Before creating a new issue, search our [open issues](https://github.com/BESSER-PEARL/BESSER/issues).  
Your issue may already be reported or in progress.

If it exists, feel free to:
- Add a **relevant comment** or clarification.
- Use **reactions** (👍 / 👎) instead of posting “+1” comments.

If you can’t find a similar issue, [create a new one](https://github.com/BESSER-PEARL/BESSER/issues/new/choose).  
The template will guide you through the process.

### 2. Writing Good Bug Reports and Feature Requests
To keep tracking easy, please:
- File **one issue per bug or feature request**.
- Include **clear reproduction steps**, screenshots, logs, or code snippets.
- Explain **why** the feature or fix is important.

The more information you include, the faster others can reproduce and fix the problem.

---

## 🍴 Setting Up Your Development Environment

If you plan to contribute code, please follow these steps to set up your workspace:

1. **Fork** the repository to your own GitHub account.  
   This creates your personal copy of the project where you can freely experiment.

2. **Clone** your fork locally:  
   ```bash
   git clone https://github.com/<your-username>/BESSER.git

---

## 💻 BESSER-Specific Contribution Guidelines

If you are contributing **code**, please follow these **repository-specific guidelines** to maintain structure and consistency.

### 1. Creating a New Code Generator
- Add new generators under the `besser/generators` directory.
- Each generator class **must inherit from** [`GeneratorInterface`](besser/generators/generator_interface.py).
- Review existing generators in this folder for implementation and best practices.

### 2. Extending the B-UML Metamodel
- Add new metamodel extensions in `besser/BUML/metamodel/`.
- Check existing submodules (e.g. `structural`, `state_machine`) for reference on structure and conventions.

### 3. Adding a New Notation
- Place new notations in `besser/BUML/notations/`.
- Use existing notations (like our PlantUML grammars) as templates.

### 4. Code Style and Agent Rules
- Follow the **code style and agent rules** defined in:
  - `.github/copilot-instructions.md`
  - `.cursor/rules/`
- These define formatting, naming, and AI-assisted coding conventions.
- All code contributions **must comply** with these rules for acceptance.

---

## 🔀 Creating Pull Requests

Ready to submit your contribution? Great! 🎉  
Please follow these steps:

1. Make your changes following the BESSER-specific guidelines above.
2. Ensure your code:
   - Passes all tests and linting checks.
   - Includes or updates relevant documentation.
   - Adds unit tests for new features or bug fixes.
3. Submit a **Pull Request (PR)** to this repository.

All pull requests are reviewed according to our project governance rules (see below).

---

## 🧭 Governance

All contributions are reviewed by project maintainers following the rules in our [GOVERNANCE.md](GOVERNANCE.md).  
This ensures fair and transparent decision-making.

---

## 💖 Thank You!

Your contributions — big or small — make this project better.  
Thank you for your time, energy, and passion for open source.
