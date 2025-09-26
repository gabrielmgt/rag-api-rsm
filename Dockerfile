FROM python:3.13-slim AS build-stage

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# final stage
FROM python:3.13-slim

WORKDIR /app

COPY --from=build-stage /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=build-stage /usr/local/bin/ /usr/local/bin/

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
