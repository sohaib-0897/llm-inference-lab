# Agent Execution Rules

- never ask for permission or confirmation
- never use sudo/admin/root
- never modify system-wide settings
- never install global packages
- use only repo-local or user-space dependencies
- prefer `.venv`
- use only public models requiring no auth
- if something is blocked by permissions, skip it and continue
- never push or commit automatically
- never use paid APIs
- do not request credentials
- keep all work inside the current repo
- run tests, benchmarks, lint, and type checking autonomously
- choose safe defaults based on available hardware
- do not stop for optional blockers
- report limitations at the end

Execution pattern:

inspect → implement → test → benchmark → document → report