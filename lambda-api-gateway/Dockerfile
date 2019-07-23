FROM amazonlinux:latest

RUN mkdir -p /mnt/app
ADD . /mnt/app
WORKDIR /mnt/app

RUN yum update -y
RUN yum install gcc -y
RUN yum install gcc-c++ -y
RUN yum install findutils -y
RUN yum install zip -y
RUN amazon-linux-extras install python3=3.6.2
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -t aws_layer/python
RUN pip3 install -r requirements_dev.txt