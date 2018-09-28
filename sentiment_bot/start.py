#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sentiment_bot import SentimentBot

needProxy = True

bot_token = ''
nn_weights_path = './model_weights_rus_tweet_1.h5py'
proxy_url = None
proxy_user = None
proxy_pass = None

if needProxy:
    proxy_url=''
    proxy_user='itusiser'
    proxy_pass='resipasutiss'

bot = SentimentBot(token=bot_token, proxy_url=proxy_url, 
                   proxy_user=proxy_user, proxy_pass=proxy_pass, 
                   nn_weights_path = nn_weights_path)
bot.start()
pass