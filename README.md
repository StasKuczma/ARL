# ARL

[PUTLab repo](https://github.com/AeroLabPUT/UAV_measurement_data)

### Co robić?
Bierzemy te dane i jakimś algorytemem trzeba ustalić które było uszkodzone, które nie.

Może być na przykład autoenkoder z pytorch'a


### Docker

To build image from Dockerfile check pytorch version!!!
go to this repo home dir.
` docker build -t arl . `

`cd docker`

`docker-compose -f docker-compose.yml up`

in other terminals.

`docker exec -it arl_pytorch bash`

This repo assumes that you have downloaded PADRE data set into home dir. Docker-compose also mounts src folder as a volume.

ARL <br>
├──docker/ <br>
├──src/ <br>
├──UAV_measurement_data/ <br> 


