FROM tensorflow/tensorflow:latest-gpu

RUN pip install poetry

ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV KERAS_BACKEND="tensorflow"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_CACHE_DIR="/p_cache/"
COPY poetry.lock pyproject.toml /app/

WORKDIR /app/

COPY ganai ganai

RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \ 
    poetry install --without dev

RUN poetry run start setup_worker ${WOR_ADDR}

ENTRYPOINT [ "poetry", "run", "start", "train", "-e", "100", '-b', '256', '-mv']