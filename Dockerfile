# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Fly will map this)
EXPOSE 8080

# Start with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "reddit:app"]