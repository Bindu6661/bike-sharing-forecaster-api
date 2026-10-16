# 1. Start with an official, lightweight Python image
FROM python:3.11-slim

# 2. Set the working folder inside the container
WORKDIR /code

# 3. Copy just your dependencies list file first
COPY ./requirements.txt /code/requirements.txt

# 4. Install all the required machine learning and API libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your application code and your trained model file into the container
COPY ./app.py /code/app.py
COPY ./best_bike_sharing_model.joblib /code/best_bike_sharing_model.joblib

# 6. Expose port 8000 so web requests can reach the app
EXPOSE 8000

# 7. Start your Uvicorn live web server when the container launches
# Replace your old CMD line with this dynamic version:
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]


