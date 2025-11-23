#!/bin/sh
exec poetry run gunicorn -w "$WORKERS" -t "${TIMEOUT:-300}" -b "0.0.0.0:$PORT" "app:create_app()"