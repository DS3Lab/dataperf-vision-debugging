FROM python:3.9-slim

RUN apt update -y && apt install gcc -y

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app/

WORKDIR /app/

ENTRYPOINT python3 create_baselines.py docker && python3 main.py docker && python3 plotter.py docker
