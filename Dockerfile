FROM python:3.10-slim

ENV POETRY_INSTALLER_PARALLEL=false

RUN pip install -U pip poetry streamlit

WORKDIR /mpx_dashboard

# Copy data, app and dependencies
COPY ./mpx_dashboard /mpx_dashboard
COPY ./data /data
COPY ./pyproject.toml /pyproject.toml
COPY ./poetry.lock /poetry.lock

RUN python3 -m pip config --user set global.timeout 150
# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-cache --only main

ENTRYPOINT [ "streamlit", "run", "mpx_dashboard/app.py" ]


