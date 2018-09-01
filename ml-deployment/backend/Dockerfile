FROM python:3.6.2

# update installation
RUN apt-get -y update
RUN apt-get -y upgrade

# create directories and copy files
RUN mkdir -p /mnt/app
ADD . /mnt/app
WORKDIR /mnt/app

# install dependencies and app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 --chdir /mnt/app/ml_app --workers 2 --reload --capture-output wsgi
