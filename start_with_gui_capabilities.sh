docker run --rm -it -d --network host --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/pyspatialkit -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
    --name pyspatialkit pyspatialkit bash
xhost +local:`sudo docker inspect --format='{{ .Config.Hostname }}' pyspatialkit`