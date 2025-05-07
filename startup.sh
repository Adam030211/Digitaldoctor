#!/bin/bash
gunicorn --bind 0.0.0.0:8000 digitaldoktor.wsgi:application
python manage.py collectstatic --noinput
python manage.py migrate