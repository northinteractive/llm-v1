# Use a TensorFlow-enabled base image
FROM tensorflow/tensorflow:latest

# Install Flask and required libraries
RUN pip install Flask

# Create a directory to store the model
RUN mkdir -p /app/models

# Copy our scripts into the Docker container
COPY train_model.py /app/
COPY app.py /app/

# Set the working directory
WORKDIR /app

# Expose the API port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
