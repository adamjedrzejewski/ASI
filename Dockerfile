FROM python:3.8-slim-buster

RUN mkdir /app
WORKDIR /app
COPY asi-project/src/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
WORKDIR asi-project
ENTRYPOINT ["kedro", "run"]

