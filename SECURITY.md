# Security Policy

## Reporting a vulnerability

If you find a security issue in memtrust, please report it privately rather than opening a public
GitHub issue.

- Preferred: use GitHub's [private vulnerability reporting](https://github.com/RudrenduPaul/memtrust/security/advisories/new)
  for this repository.
- Alternative: email the maintainers directly. Include a description of the issue, steps to
  reproduce, and the potential impact. If you can suggest a fix, that's welcome but not required.

Please do not include real vendor API keys, real credentials, or real personal data in any report
or reproduction steps -- use placeholder values.

## What's in scope

- The `memtrust` harness, CLI, adapters, eval runners, and scoring pipeline in `src/memtrust/`.
- The static leaderboard site in `leaderboard/`.
- The GitHub Actions workflow in `.github/workflows/`.

## What's out of scope

- Vulnerabilities in the vendor backends themselves (MemPalace, Mem0, Zep/Graphiti, OpenViking) --
  report those to the respective projects.
- Vulnerabilities that require a leaked, malicious, or attacker-controlled API key already granted
  to memtrust -- memtrust trusts credentials it's explicitly configured with, the same as any CLI
  tool that reads a token from an environment variable.

## Response

We aim to acknowledge a report within 5 business days and to have a fix or a mitigation plan
within 30 days for confirmed issues, faster for anything actively exploitable. Credit is given in
the fix's changelog entry unless you ask not to be named.

## Supported versions

memtrust is pre-1.0 (`0.x`). Security fixes land on `main` and the latest released version only --
there is no long-term-support branch at this stage.
