#!/bin/sh
if [ "$FLASK_ENV" = "development" ]; then
    poetry run python ./app.py
else
    poetry run gunicorn --bind 0.0.0.0:$HANDWRITING_PORT app:app
fi