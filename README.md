# ARL

[PUTLab repo](https://github.com/AeroLabPUT/UAV_measurement_data)

### Co robić?
Bierzemy te dane i jakimś algorytemem trzeba ustalić które było uszkodzone, które nie.


UPDATE 2 Grudzień

[x] Autoencoder <br/>
[x] Isolation Forest
<br/>Chat GPT proponuje <br/>
[ ] Transformacja sygnałów + FFT <br/> 
[ ] LSTM Autoencoder <br/>
[ ] Dynamiczne klastry <br/>


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


