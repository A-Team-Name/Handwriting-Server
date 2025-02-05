FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    curl && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    pip install poetry && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/* /usr/share/info/*


WORKDIR /app

COPY pyproject.toml poetry.lock app.py load_model.py entrypoint.sh /app/

COPY /models /app/models

RUN poetry config virtualenvs.create false && \
    poetry install && \
    rm -rf /root/.cache

# RUN poetry run python3 load_model.py

ENTRYPOINT ["/bin/sh", "./entrypoint.sh"]
