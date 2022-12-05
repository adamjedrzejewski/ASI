FROM ubuntu:22.10

RUN apt -y update && apt -y install python3

COPY main.py /my-scripts/main.py

ARG CONFIG_CONTENTS=aaa
RUN echo -n "$CONFIG_CONTENTS" > /config

ENTRYPOINT ["python3"]

CMD ["my-scripts/main.py"]

