# Lambda instructions

- Run docker AmazonLinux container

        python docker.py -a start

- Rebuild docker AmazonLinux container image

        python docker.py -a rebuild

- Test

        python docker.py -a ssh
        pytest --capture=no --verbose tests

- Add dependencies locally

        python docker.py -a ssh
        pip3 install -r [dependency] -t aws_layer/python

- Zip code

        python docker.py -a ssh
        find -name "*.so" | xargs strip
        python3 archive.py
        [upload Archive.zip to Lambda]

- Zip dependencies as AWS layer
  
        python docker.py -a ssh
        find -name "*.so" | xargs strip
        python3 aws_layer_compress.py
        [upload aws_layer/python.zip to Lambda]
        [select current dependency version in Lambda function management that uses the layer]

- (Optional) upload Archive.zip to S3 and then load to Lambda

        aws s3 [--profile profile_name] cp Archive.zip s3://bucket_name