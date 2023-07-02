FROM huggingface/transformers-pytorch-gpu:4.29.2
RUN pip install einops

WORKDIR /var/models/
COPY . /usr/scripts/chat/

ENV TRANSFORMERS_OFFLINE=1
EXPOSE 8080
CMD [ "python3", "/usr/scripts/chat/chat_server.py" ]
