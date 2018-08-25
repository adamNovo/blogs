# Lambda instructions

- Run docker AmazonLinux container

        python docker.py -a start

- Develop Lambda function

        python docker.py -a ssh
        python3 lambda_function.py

- (Optional) add dependencies locally

        pip3 install [dependency] -t .

- zip code and dependecies

        python docker.py -a ssh
        find -name "*.so" | xargs strip
        python3 archive.zip

- [upload Archive.zip to S3 and then load to Lambda]

        aws s3 [--profile profile_name] cp Archive.zip s3://bucket_name