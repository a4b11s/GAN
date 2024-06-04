FROM tensorflow/tensorflow:latest-gpu

RUN pip install poetry

ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV KERAS_BACKEND="tensorflow"
ENV POETRY_VIRTUALENVS_CREATE=false

COPY poetry.lock pyproject.toml /app/

WORKDIR /app/

COPY ganai ganai

RUN poetry install --without dev

ENTRYPOINT [ "poetry", "run", "start", "-t", "-e", "100"]