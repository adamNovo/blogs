# ML App Deployment

## Run docker app locally

        cd backend
        python docker.py -a start

## Run tests (different terminal)

        cd backend
        source activate [conda_env_name]
        pip install -r requirements.txt
        pytest --capture=no -vv

## Stop docker app and cleanup

        cd backend
        python docker.py -a rm