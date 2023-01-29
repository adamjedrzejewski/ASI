FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl gcc
RUN apt-get install libgomp1
RUN mkdir /app
WORKDIR /app
COPY asi-project/src/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
WORKDIR asi-project
ENTRYPOINT ["kedro", "run", "--pipeline", "sprint_two_housing"]

