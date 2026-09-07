# 🔒 Security Policy

## 🔑 Credential & Key Safety

- **Automated Workspace Protection (`.gitignore`):** On first container startup, a secure `.gitignore` is automatically created in `/workspace/` to prevent users from accidentally committing API keys (`.env`), datasets (`data/`), model weights (`models/`), or runtime logs (`logs/`).
- **Never commit `.env` files**: All private LLM keys, cloud tokens, and database passwords must remain in `.env`, which is strictly excluded via `.gitignore` and `.dockerignore`.
- **Use `.env.example` as a template**: Only commit placeholder strings in `.env.example`.
- **Automated Git Setup (`tensorbox git-init`):** Users can run `tensorbox git-init` in the VS Code terminal to automatically verify and enforce repository isolation before pushing to their personal GitHub account.

---

## 🛡️ Reporting a Vulnerability

If you discover a security vulnerability or potential credential leak:
1. Please do not open a public issue.
2. Email the maintainers directly with details and replication steps.
3. A patched release will be issued promptly.
