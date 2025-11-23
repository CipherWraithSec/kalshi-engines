FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm
COPY src ./src
COPY models ./models
COPY data_samples ./data_samples
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "kalshi_engines", "--engine", "weather-oil", "--demo"]