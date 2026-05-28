# ds-code

Minimal Python skeleton for a DeepSeek-TUI port.

## Run

- Create a venv and install dependencies.
- Start the TUI:
  - ds-code

## Prompt routing

Prefix prompts to invoke tools directly:

- tool:my_tool arg1 arg2
- shell:ls -la
- mcp:server::tool {"key":"value"}
