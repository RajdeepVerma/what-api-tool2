ARG FUNCTION_DIR="/function"

FROM python:3.8

RUN pip install --upgrade pip

RUN apt-get update &&\
    apt -y install libsndfile1 &&\
    apt install -y software-properties-common &&\
    apt install -y python3-pip

COPY ./ /${FUNCTION_DIR}/

WORKDIR /${FUNCTION_DIR}/

RUN pip3 install -r ${FUNCTION_DIR}/requirements.txt --no-cache-dir

EXPOSE 5000

ENTRYPOINT [ "python3" ]


CMD [ "app.py" ]