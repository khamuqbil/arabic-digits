#!/bin/bash

# Startup script for Azure App Service
echo "Starting Arabic Digits Recognition App..."

# Install dependencies if needed
pip install -r requirements.txt

# Start the application with gunicorn
cd /home/site/wwwroot || exit
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
gunicorn --bind=0.0.0.0:80 --workers=2 --timeout=600 demo.main:app

