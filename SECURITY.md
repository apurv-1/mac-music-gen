# Security

## Reporting a problem

Please report security problems privately through GitHub's "Report a vulnerability" button on the repository's Security tab, not in a public issue.
Expect a reply within a week.

## What mac-music-gen does to stay safe

- It installs ACE-Step from one audited commit, and `uv.lock` pins that tarball's hash.
- It downloads model weights from one pinned Hugging Face revision, and only when you run `mac-music-gen fetch`.
- Generation runs in a separate process with Hugging Face offline mode on, so the model cannot download anything.
- ACE-Step loads its model with `trust_remote_code`, but it first replaces the model's Python files with copies from its own pinned package.
  AUDIT.md explains this and the other findings.

If you change either pin, re-run the review described in AUDIT.md.
