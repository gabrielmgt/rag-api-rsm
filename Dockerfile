FROM python:3.13-slim 

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .

RUN pip install -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
