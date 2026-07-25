# description

## install & use

```shell
# install conda (if it is not installed)
make conda-install
make conda-setup

# install dependencies
make env-create
make env-poetry-install

# list cameras
bin/run python3 src/app.py cameras

# run
bin/run python3 src/app.py detect --camera 4
bin/run python3 src/app.py depth --camera 4
bin/run python3 src/app.py segmentation --camera 4
```
