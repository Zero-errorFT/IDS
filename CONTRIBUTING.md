# Contributing to CANARY

Thanks for wanting to help! Here's how:

## 📋 Before You Start

1. Fork the repo (click Fork on GitHub)
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/canary-nids.git`
3. Create a branch: `git checkout -b fix/my-bug` or `git checkout -b feature/my-feature`

## ✍️ Code Style

- Follow PEP 8 (Python style guide)
- Add comments for complex code
- Use descriptive variable names

## 🧪 Testing

Before submitting:
```bash
# Run tests
python -m pytest tests/

# Check code style
pip install black flake8
black .
flake8 .