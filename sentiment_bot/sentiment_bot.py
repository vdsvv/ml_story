#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sentiment_nn import SentimentNN 
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

class SentimentBot:
    def __init__(self, token, nn_weights_path, proxy_url = None, proxy_user = None, proxy_pass = None):
        self.token = token
        self.proxy_url = proxy_url
        self.proxy_user = proxy_user,
        self.proxy_pass = proxy_pass,
        self.nn_weights_path = nn_weights_path
        self.REQUEST_KWARGS = None
        if self.proxy_url != None:
             self.REQUEST_KWARGS = {
                'proxy_url': proxy_url,
                'urllib3_proxy_kwargs': {
                    'username': proxy_user,
                    'password': proxy_pass
                }
            }
        pass
    
    def start(self):
        print('>>SentimentNN')
        self.nn = SentimentNN()
        self.nn.createModel()
        self.nn.loadWeights(self.nn_weights_path)
        prediction = self.nn.predict('привет')
        print('<<SentimentNN')
        updater = Updater(token = self.token, request_kwargs = self.REQUEST_KWARGS) # Токен API к Telegram
        dispatcher = updater.dispatcher
        start_command_handler = CommandHandler('start', self.startCommand)
        text_message_handler = MessageHandler(Filters.text, self.textMessage)
        dispatcher.add_handler(start_command_handler)
        dispatcher.add_handler(text_message_handler)
        updater.start_polling(clean=True)
        updater.idle()
        pass
    
    def stop(self):
        pass
    
    def startCommand(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text='Привет ! ! !')
    
    def textMessage(self, bot, update):
        prediction = 0
        try:
            prediction = self.nn.predict(update.message.text)[0][0]
        except Exception as e:
            print(e)
            pass
        response = 'Ваше сообщение позитивное на: {0:.0f}% '.format(prediction * 100)
        bot.send_message(chat_id=update.message.chat_id, text=response)
