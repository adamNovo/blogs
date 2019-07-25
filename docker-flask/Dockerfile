FROM python:3.6

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

CMD GUNICORN_CMD_ARGS="--bind=0.0.0.0:5000 --workers=1 --chdir /mnt/app/package --reload True --log-level debug --capture-output" gunicorn wsgi
