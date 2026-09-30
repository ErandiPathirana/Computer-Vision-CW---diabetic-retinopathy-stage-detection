# Container for the OcuGrade web app.
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY webapp ./webapp
COPY models ./models
ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080
CMD ["python", "-m", "webapp.server"]
