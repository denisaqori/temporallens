@AGENTS.md

## Claude-specific notes

The shared operating rules are in `AGENTS.md` (imported above). Only Claude-specific guidance
belongs here.

- **Rooting.** When run from the Claude desktop app, the session may be rooted at the parent
  `…/TemporalLens/` rather than at this repo. Until the project folder is pointed at
  `…/TemporalLens/temporallens`, use absolute paths under the repo and never write relative to the
  parent. Confirm with `git rev-parse --show-toplevel`.
- **Editing Word planning docs.** `planning-documents/` is a separate sibling repo, outside this one.
  Edit `.docx` files via the `docx` skill (unzip → edit `word/document.xml` → rezip → validate), and
  keep a backup before overwriting. Those files are not tracked by this repository.
- **Persistent memory** lives outside the repo (see the memory directory in the system prompt); do
  not store project state there that belongs in `docs/project/STATUS.md` or `DECISIONS.md`.
