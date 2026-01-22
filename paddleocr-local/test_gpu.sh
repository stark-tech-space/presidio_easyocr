#!/bin/bash
sudo docker run --gpus all --rm paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5 bash -c 'python -c "import paddle;paddle.device.set_device(\"gpu:0\");x=paddle.randn([1000,1000]);y=paddle.matmul(x,x);print(\"GPU:\",paddle.device.cuda.get_device_name(0));print(\"OK\")"'
