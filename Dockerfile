# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app -- this was included in the template, but I don't think it's necessary here
# WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 80 (or the port your application uses)
EXPOSE 80

# Set environment variables (if any)
ENV ENVIRONMENT=development

# Command to run your application
CMD ["python", "game.py"]