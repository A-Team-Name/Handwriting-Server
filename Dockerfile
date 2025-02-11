FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    curl \
    wget && \
    wget -O apl.deb https://www.dyalog.com/uploads/php/download.dyalog.com/download.php?file=19.0/linux_64_19.0.50027_unicode.x86_64.deb && \
    dpkg -i apl.deb && \
    rm apl.deb && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    pip install --upgrade pip && \
    pip install poetry && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/* /usr/share/info/*


WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry config virtualenvs.create false && \
    poetry install && \
    rm -rf /root/.cache

COPY app.py app.py
COPY load_model.py load_model.py
COPY entrypoint.sh entrypoint.sh
COPY models/ /app/models


# RUN poetry run python3 load_model.py

ENTRYPOINT ["/bin/sh", "./entrypoint.sh"]
