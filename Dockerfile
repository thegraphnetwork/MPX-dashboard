FROM python:3.10-slim

ENV POETRY_INSTALLER_PARALLEL=false

RUN pip install -U pip poetry streamlit

WORKDIR /mpx_dashboard

# Copy data, app and dependencies
COPY ./mpx_dashboard /mpx_dashboard
COPY ./data /data
COPY ./pyproject.toml /pyproject.toml
COPY ./poetry.lock /poetry.lock

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only main

ENTRYPOINT [ "streamlit", "run", "mpx_dashboard/app.py" ]


