FROM python:3.8

WORKDIR /saver

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install transformers safetensors accelerate --no-cache-dir

ENV MODEL_FOLDER=/krea-models 

COPY . .

CMD python3 decode_credentials.py && python3 new-model-saver.py
