# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR ./RLEngine
ENV PYTHONPATH=.
ENV PATH=$PATH:.:/RLEngine
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
COPY pygad.py /usr/local/lib/python3.8/site-packages/pygad/pygad.py
COPY fmin.py /usr/local/lib/python3.8/site-packages/hyperopt/fmin.py
CMD ["/bin/bash", "-c", "./comando.sh"]
#CMD ["python", "./RLEngine/RunFromEnv.py"]
#CMD ["python", "./checkversion.py"]
