FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get -y install python3.11 python3-pip

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY . .

RUN pip install poetry

RUN poetry env use 3.11

RUN poetry config virtualenvs.create false

RUN poetry install

RUN poetry run python load_model.py

CMD ["poetry", "run", "python", "app.py"]
