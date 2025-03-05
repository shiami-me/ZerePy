#!/bin/bash
set -e

# Run any potential migrations or setup commands here
# Example: poetry run python -m alembic upgrade head

# Execute the CMD from the Dockerfile, which runs your application
exec poetry run python main.py "$@"
