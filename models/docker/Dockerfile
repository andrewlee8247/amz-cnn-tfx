FROM tensorflow/tensorflow:1.9.0-py3

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install p7zip

COPY . /app

WORKDIR /app

RUN pip install -r /app/models/requirements.txt --no-cache

ENTRYPOINT ["python", "/app/models/model.py"]