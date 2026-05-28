# Tasks

## Overview
This layer provides a thin wrapper around job management. It is used by the runtime API and web server to enqueue and update jobs.

## Files
- [manager.py](manager.py): TaskManager wrapper around JobManager.
- [__init__.py](__init__.py): Package marker.

## How It Fits Together
- [runtime/api.py](../runtime/api.py) and [web/server.py](../web/server.py) use TaskManager to create and update jobs.
- Job state is owned by [core/jobs.py](../core/jobs.py).
