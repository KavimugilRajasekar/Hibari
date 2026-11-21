# Contributing to Hibari

Thank you for your interest in contributing to Hibari!

Hibari is an open-source cognitive LLM architecture and language model project designed and maintained by [Kavimugil Rajasekar](https://kavimugilrajasekar.github.io).

---

## Getting Started

1. Fork the repository on GitHub: [https://github.com/KavimugilRajasekar/Hibari](https://github.com/KavimugilRajasekar/Hibari)
2. Clone your fork locally:
   ```bash
   git clone https://github.com/KavimugilRajasekar/Hibari.git
   cd Hibari
   ```
3. Set up the Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install datasets tokenizers gguf pytest
   ```
4. Create a dedicated feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. Make your modifications and verify all tests pass:
   ```bash
   pytest
   ```
6. Commit your changes following standard commit conventions.
7. Push to your branch and open a Pull Request against `main`.

---

## Code Guidelines

- Adhere to PEP 8 standards.
- Use explicit type annotations for function signatures.
- Write docstrings for all modules, classes, and public functions.
- Do not add emoji characters in markdown files, docstrings, or console logs.
- Write automated tests in `hibari/tests/` for any new models, transforms, or tools.

---

## Commit Message Guidelines

Use structured, descriptive commit messages:
- `feat: add kv cache support to attention layer`
- `fix: correct tokenizer special token offset`
- `docs: update mathematical formulations in Docs/04_MODEL_DESIGN.md`
- `test: add unit test for nucleus sampling bounds`

---

## Author & Contact

- Author: Kavimugil Rajasekar
- GitHub: [KavimugilRajasekar](https://github.com/KavimugilRajasekar)
- Portfolio: [https://kavimugilrajasekar.github.io](https://kavimugilrajasekar.github.io)
- Repository: [https://github.com/KavimugilRajasekar/Hibari](https://github.com/KavimugilRajasekar/Hibari)
