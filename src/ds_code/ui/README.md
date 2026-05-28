# UI

## Overview
Optional UI shells for local testing. The current primary UI is the web server in [web/server.py](../web/server.py).

## Files
- [tui.py](tui.py): Textual-based terminal UI.
- [tk_app.py](tk_app.py): Tkinter window UI.
- [__init__.py](__init__.py): Package marker.

## How It Fits Together
- Both UIs call [runtime/api.py](../runtime/api.py) directly.
- These are optional and can be left unused when running the web UI.
