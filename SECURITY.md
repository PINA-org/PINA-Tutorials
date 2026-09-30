# Security Policy

This repository contains only tutorial material: Jupyter notebooks, the images
they embed, and the datasets they ship. It is executed by CI and published as
static HTML, so the attack surface is a notebook that downloads or unpickles
something at build time.

**Report a vulnerability in the PINA library itself in the main repository:**
[PINA-org/PINA](https://github.com/PINA-org/PINA). Use the
["Report a vulnerability"](https://github.com/PINA-org/PINA/security/advisories/new)
button under the "Security" tab; it opens a private channel with the
maintainers.

## Reporting a vulnerability here

If the issue is specific to this repository — for example a tutorial that
downloads from an untrusted host, or a shipped dataset that is unexpectedly a
pickle — report it the same way, in the main repository, and mention
`PINA-Tutorials` in the description. There is no separate security release
cadence here.

## Hardening notes for tutorial authors

- Prefer plain formats (`.mat`, `.npz`, `.txt`) over pickles. A `.pt` file
  loaded with `torch.load` executes arbitrary code, so a shipped `.pt` must be
  treated as trusted and must not be silently replaced.
- Download datasets from a `raw.githubusercontent.com` URL pointing at a
  specific branch or tag, never from a redirecting or short-lived host.
- Do not commit credentials, tokens, or private endpoints.
