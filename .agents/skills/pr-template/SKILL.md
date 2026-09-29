---
name: pr-template
description: Fills eagle-eye's required pull-request template when opening a PR through GitHub CLI or a connected GitHub tool. Trigger this when the user asks to open/create/submit a pull request or push changes up for review in this repository.
---

# PR Template Filler

eagle-eye requires every PR to follow `.github/PULL_REQUEST_TEMPLATE.md`. This skill fills that template from the real state of the branch instead of writing a generic summary.

## When this applies

Any time you're about to open a pull request through GitHub CLI or a connected GitHub tool in this repo. If `.github/PULL_REQUEST_TEMPLATE.md` doesn't exist, fall back to a plain description.

## Steps

1. **Read the template**: `.github/PULL_REQUEST_TEMPLATE.md`. Use its exact section headings — don't paraphrase or reorder them.

2. **Determine the base branch** from an explicit user choice, an existing upstream/PR configuration, or the Gitflow rules in `git-workflow.md`. Feature branches normally target `develop`; release and hotfix branches follow their documented destinations. Do not silently assume `main`.

3. **Gather branch facts** before writing anything:
   - `git log <base-branch>..HEAD --oneline` — every commit going into this PR, not just the latest one.
   - `git diff <base-branch>...HEAD --stat` — which files/packages changed.

4. **Fill "Description"** with why the change was made (1-3 sentences), based on the actual commits/diff, not a restatement of file names.

5. **Fill "Type of Change"** by checking the box(es) matching the Conventional Commit `type` prefixes actually present in the commit log (e.g. commits starting `feat:` check `feat`, `fix:` checks `fix`). Check every type that appears at least once; leave the rest unchecked.

6. **Fill "Related Issue / Ticket"** only if an issue/ticket number was mentioned in the conversation or in a commit footer (e.g. `Fixes #123`, `Closes #5`). Otherwise leave the placeholder as-is — don't invent an issue number.

7. **Fill "Key Changes"** (Added/New, Updated/Changed, Fixed) as short bullets derived from the diff. Omit a subsection entirely if nothing in the diff fits it — don't leave a bullet like "N/A" under an empty heading.

8. **Fill "Pre-Commit Checklist" honestly** — check a box **only if its condition was actually verified**:
   - Check "Ran `pnpm lint`" only if you ran it (or `pnpm --filter <pkg> lint`) this session and it exited clean.
   - Check "Ran `pnpm typecheck`" only if you ran it this session and it exited clean.
   - Check "Ran `pnpm test`" only if you ran it this session and it passed.
   - Check the style-guide box only after reviewing the change against `AGENTS.md`; the repository template may still mention `CLAUDE.md`, but `AGENTS.md` is the Codex instruction source.
   - Check the documentation box only after reviewing whether relevant documentation was updated.
   - Run checks required by `AGENTS.md` before opening the PR. If a check is not required or the user explicitly asks to skip it, leave its box unchecked rather than claiming it passed.

9. **Open the PR** with the filled template as the body via `gh pr create --body-file <body-file>` or an available connected GitHub PR tool, then report the PR URL.

## Why honesty on the checklist matters

The checklist is what a reviewer trusts to skip re-running checks locally. A checked box that wasn't actually verified is worse than an unchecked one — it tells the reviewer something is safe when it isn't.
