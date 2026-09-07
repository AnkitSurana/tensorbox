# 🤝 Contributing Guidelines

Contributions are welcome! Follow these steps to contribute to the devbox environment:

## 🛠️ Development Workflow

1. Fork and clone the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Make your changes in code, Docker configurations, or documentation.
4. Run the automated test suite to ensure no regressions:
   ```bash
   pytest tests
   ```
5. Test docker compose configuration:
   ```bash
   docker compose config -q
   ```
6. Commit and push your changes, then submit a Pull Request.

---

## 📋 PR Checklist
- [ ] Code follows standard PEP 8 formatting.
- [ ] New features include corresponding unit tests in `tests/`.
- [ ] Documentation updated in `README.md` or `docs/` if modifying ports or tools.
