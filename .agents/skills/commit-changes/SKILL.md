---
name: commit-changes
description: Commit requested changes in this Python trajectory-forecasting repository using Conventional Commits and scoped verification. Use when the user asks to commit, check in, or save changes to Git. A commit request does not authorize training or pushing.
---

# Commit Changes

Apply Conventional Commits v1.0.0 in this repository. Read the applicable `AGENTS.md`; consult other Git workflow documentation only if it exists. Preserve the official model baseline and unrelated user changes.

## Workflow

### 1. Inspect and determine scope

- Inspect `git status`, staged and unstaged diffs, and recent `git log`.
- Determine which files belong to the requested commit.
- Existing staged changes are not implicit permission to commit them.
- If the requested scope is ambiguous, ask before staging or committing.

### 2. Verify proportionally to the changes

Use the existing `.venv/bin/python` when available.

- For changed Python files, check syntax and run relevant existing tests.
- Inspect the tests and use the runner the repository actually supports. Do not assume pytest, linting or type checking is configured.
- For JSON configurations, verify parsing and relevant configuration invariants without starting training.
- For plotting changes, use existing artifacts for a short generation check and inspect the output when feasible. Do not overwrite user artifacts without authorization.
- Markdown-only changes do not require Python tests, but still review the diff and referenced paths.
- Check the intended changes for whitespace errors using `git diff --check` and `git diff --cached --check` with specific paths.
- Separate unrelated existing failures from failures caused by the intended changes.
- Do not claim unavailable or unperformed checks passed.

If verification fails:

- Fix only issues within the requested scope, then run the relevant checks again.
- Ask for direction if a fix would change the algorithm, baseline, environment or unrelated code.
- Do not commit code with unresolved relevant failures.
- Never use `--no-verify` to bypass hooks.

Do not start training, full test-set evaluation, dependency installation or expensive experiments merely to prepare a commit. Request explicit authorization if such work is genuinely necessary.

### 3. Prepare the commit message

Use Conventional Commits:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Rules:

- Choose the type matching the actual change:
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, or `revert`.
- Use an English imperative description, such as `add`, `fix` or `change`.
- Do not capitalize the first word unless it is a proper noun or symbol.
- Do not end the description with a period.
- Include a scope when the change is localized to a module.
- Mark breaking changes with `!` before the colon or a
  `BREAKING CHANGE:` footer.

Examples:

```text
feat(visualization): add IMPTC loss comparison
fix(checkpoint): restore optimizer state on resume
docs(experiments): explain sharpness normalization
chore(skills): adapt commit workflow for Python repository
```

When requested work includes unrelated concerns, explain the proposed split before making separate commits. Do not include unrelated pre-staged changes.

### 4. Stage only intended files

- Stage specific files by name.
- Never use `git add .` or `git add -A`.
- Preserve unrelated staged and unstaged changes.
- Do not force-add ignored data, checkpoints, results, slides,
  virtual environments or credentials.
- If the user requests ignored artifacts, inspect the exact targets
  and sizes and explain constraints before staging.
- Do not introduce Git LFS unless explicitly requested.

### 5. Commit and report

- Review the intended diff immediately before committing.
- Commit only authorized files.
- If unrelated files are already staged, use a path-scoped commit
  where appropriate or ask for direction.
- A heredoc is optional; ensure the commit message is correctly formatted.
- After committing, run `git status`.

Report:

- Commit hash and message.
- Files or scope included.
- Checks performed and any limitations.
- Remaining staged, unstaged or untracked changes.

Do not force a clean worktree by discarding or committing unrelated work.

### 6. Handle failed attempts safely

- Never amend unless the user explicitly asks.
- If a hook rejects an attempt, check whether a commit was actually
  created before retrying.
- Fix only in-scope issues, re-stage the intended files and retry
  without `--amend`.
- If a fix is required after a successful commit, create a new commit
  unless the user explicitly requests an amend.

## Repository boundaries

- Commit locally unless the user separately requests a push.
- Do not implicitly open a pull request.
- Preserve upstream `README.md`.
- Use `README_LOCAL.md` for local setup and execution instructions.
- Use `MIDTERM_NOTES.md` for academic scope and experiment planning.
- Do not change architecture, loss, optimizer, preprocessing,
  official metrics or important hyperparameters merely to prepare a commit.
- Conventional Commits is a message convention here. Do not claim
  semantic-release or Gitflow automation exists without verifying it.