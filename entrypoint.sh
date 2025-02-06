#!/bin/sh
if [ "$FLASK_ENV" = "development" ]; then
    flask --app app run --debug --host=0.0.0.0 --port=$HANDWRITING_PORT
else
    gunicorn --bind 0.0.0.0:$HANDWRITING_PORT app:app
fi