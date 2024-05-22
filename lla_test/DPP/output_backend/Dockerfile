FROM python:3.9-slim
WORKDIR /app

COPY main_api.py /app
COPY pydantic_classes.py /app
COPY sql_alchemy.py /app

RUN pip install requests==2.31.0
RUN pip install fastapi==0.110.0
RUN pip install pydantic==2.6.3
RUN pip install uvicorn==0.28.0
RUN pip install SQLAlchemy==2.0.29
RUN pip install httpx==0.27.0

EXPOSE 8000
CMD ["python", "main_api.py"]
