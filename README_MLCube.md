### Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the vision selection repo
git clone https://github.com/DS3Lab/dataperf-vision-debugging && cd ./dataperf-vision-debugging
```

### Execute single tasks with MLCube

```bash
# Download and extract dataset
mlcube run --task=download -Pdocker.build_strategy=always

# Create baselines submission (skip this if you have already done so, or you want to evaluate your submissions only)
mlcube run --task=create_baselines -Pdocker.build_strategy=always

# Run evaluation
mlcube run --task=evaluate -Pdocker.build_strategy=always

# Run plotter
mlcube run --task=plot -Pdocker.build_strategy=always
```

### Execute all tasks with MLCube

```
mlcube run --task=download,create_baselines,main,plot -Pdocker.build_strategy=always
```
