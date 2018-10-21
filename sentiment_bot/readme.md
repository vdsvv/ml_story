# Telegram bot with sentiment analysis recurrent neural network.

### Bot name
@ImitatioBot

### Bot View

![](preview_sentiment_bot.png)

### Deploy
```shell
#[step 1]
docker build -t sentiment_bot:v1.0-no-proxy .

#[step 2]
docker run sentiment_bot:v1.0-no-proxy

#[step 3]
docker tag sentiment_bot:v1.0-no-proxy gcr.io/telegrambot-1-215117/sentiment_bot:v1.0-no-proxy

#[step 4]
#https://console.cloud.google.com/gcr/images/telegrambot-1-215117?project=telegrambot-1-215117
docker push gcr.io/telegrambot-1-215117/sentiment_bot:v1.0-no-proxy

#[step 5]
#https://console.cloud.google.com/compute/instances?project=telegrambot-1-215117
gcloud beta compute instances create-with-container sentiment-bot --zone us-central1-a --container-image=gcr.io/telegrambot-1-215117/sentiment_bot:v1.0-no-proxy --machine-type=g1-small
```

### Documentation
https://docs.docker.com/get-started/

https://cloud.google.com/container-registry/docs/quickstart

https://cloud.google.com/container-registry/docs/pushing-and-pulling

https://console.cloud.google.com/compute/instances?project=telegrambot-1-215117

https://cloud.google.com/compute/docs/containers/deploying-containers

### Other
```
gcloud beta compute ssh telegrambot-1 --container gcr.io/telegrambot-1-215117/tb-1-image
gcloud builds submit --tag gcr.io/telegrambot-1-215117/sentiment_bot:v1.0-no-proxy .
```