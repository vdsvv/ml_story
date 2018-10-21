#! /usr/bin/env python
# -*- coding: utf-8 -*-

#Мир спасёт красота.
#Нестерпимая же вонь из распивочных, которых в этой части города особенное множество, и пьяные, поминутно попадавшиеся, несмотря на буднее время, довершили отвратительный и грустный колорит картины.
#Кстати, он был замечательно хорош собою, с прекрасными темными глазами, темно-рус, ростом выше среднего, тонок и строен.
#Работа не волк, в лес не убежит.
#Мне нравятся котики.
#Собака друг человека
#Завтра мы все умрём!

from sentiment_bot import SentimentBot

needProxy = False

bot_token = ''
#https://drive.google.com/open?id=1FZSxF9Bt8n47LjeYbeHcb9hb4MfFi43V
nn_weights_path = './model_weights_rus_tweet_1.h5py'
proxy_url = None
proxy_user = None
proxy_pass = None

if needProxy:
    proxy_url=''#socks5://gttjghf.com:55655/
    proxy_user='itusiser'
    proxy_pass='resipasutiss'

bot = SentimentBot(token=bot_token, proxy_url=proxy_url, 
                   proxy_user=proxy_user, proxy_pass=proxy_pass, 
                   nn_weights_path = nn_weights_path)
bot.start()
pass