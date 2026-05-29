# Tests

## Setup
Install dev dependencies:

```
pip install -e ".[dev]"
```

Set required environment variables (recommended: use .env.test loaded by pytest-dotenv):

Create `.env.test` in the repo root:

```
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_FIM_MODEL=deepseek-v4-flash
DEEPSEEK_TRANSLATE_MODEL=deepseek-v4-flash
```

If you prefer exporting variables manually, use:

```
export DEEPSEEK_API_KEY="your_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
```

Optional:

```
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_FIM_MODEL="deepseek-v4-flash"
export DEEPSEEK_TRANSLATE_MODEL="deepseek-v4-flash"
```

## Run integration tests
Run all integration tests:

```
PYTHONPATH=src pytest -q -m integration
```

Run a single file:

```
PYTHONPATH=src pytest -q tests/test_deepseek_integration.py
```

Run a single test by name:

```
PYTHONPATH=src pytest -q tests/test_deepseek_integration.py -k test_list_models_live
```
