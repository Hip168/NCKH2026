# Một image dùng chung cho mọi dịch vụ Python: ngân hàng, điều phối, consumer, producer.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1
WORKDIR /app

# Bản torch chỉ dùng CPU, nhẹ hơn nhiều so với bản có CUDA
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ha-tang.txt ./
RUN grep -v -E "^torch==|^pytest" requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt -r requirements-ha-tang.txt

COPY src ./src
COPY scripts ./scripts
COPY fl_app ./fl_app
COPY data/splits ./data/splits
