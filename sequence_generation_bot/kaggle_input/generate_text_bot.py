#! /usr/bin/env python
# -*- coding: utf-8 -*-
#https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#post-a-text-message-with-html-formatting
#from sequence_generator_nn import SequenceGeneratorNN 
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram as tg
#from service_bot import ServiceBot

class GenerateTextBot:
    def __init__(self, neural_net, params, service_bot):
        self.sb = service_bot
        self.__print('👉 gtbot::__init__')
        self.nn = neural_net
        self.__ps = params
        self.token = self.__ps['work_bot_token']
        self.need_proxy = self.__ps['bot_need_proxy']
        #bot_proxy_url='socks5://iork.com:55655/',
        self.proxy_url = '{}://{}/'.format(self.__ps['bot_proxy_proto'], self.__ps['bot_proxy_url']) if self.need_proxy else None
        self.proxy_user = self.__ps['bot_proxy_user'] if self.need_proxy else None
        self.proxy_pass = self.__ps['bot_proxy_pass'] if self.need_proxy else None
        self.nn_weights_path = self.__ps['path_models'] + self.__ps['default_model_name'] + self.__ps['default_model_ext']
        self.REQUEST_KWARGS = None
        if self.proxy_url != None:
            self.REQUEST_KWARGS = {
                'read_timeout': 20,
                'connect_timeout': 10,
                'proxy_url': self.proxy_url,
                'urllib3_proxy_kwargs': {
                    'username': self.proxy_user,
                    'password': self.proxy_pass
                }
            }
        #else:
        #    self.REQUEST_KWARGS = {
        #        'read_timeout': 20,
        #        'connect_timeout': 10,
        #    }
        self.__print('👈 gtbot::__init__')
    
    def start(self):
        self.__print('👉 gtbot::start')
        updater = Updater(token = self.token, request_kwargs = self.REQUEST_KWARGS) # Токен API к Telegram
        dispatcher = updater.dispatcher
        start_command_handler = CommandHandler('start', self.startCommand)
        temperature_command_handler = CommandHandler('temp', self.temperatureCommand)
        words_command_handler = CommandHandler('words', self.wordsCommand)
        begin_words_command_handler = CommandHandler('bwords', self.beginWordsCommand)
        text_message_handler = MessageHandler(Filters.text, self.textMessage)
        dispatcher.add_handler(start_command_handler)
        dispatcher.add_handler(temperature_command_handler)
        dispatcher.add_handler(words_command_handler)
        dispatcher.add_handler(begin_words_command_handler)
        dispatcher.add_handler(text_message_handler)
        updater.start_polling(clean=True)
        self.__print('👈 gtbot::start')
        updater.idle()
        pass
    
    def stop(self):
        pass
    
    def startCommand(self, bot, update):
        #bot.send_message(chat_id=update.message.chat_id, text='Команды: \n/temp\n/words')
        pass

    def temperatureCommand(self, bot, update):
        try:
            command_param = update.message.text.rsplit(None, 1)
            command_param = float(command_param[-1])
            self.nn.setTemperature(command_param)
            bot.send_message(chat_id=update.message.chat_id, text='☘️ Успешно ☘️')
        except Exception as e:
            bot.send_message(chat_id=update.message.chat_id, text='🔥 Ошибка 🔥')
            print(e)
            pass
    
    def wordsCommand(self, bot, update):
        try:
            command_param = update.message.text.rsplit(None, 1)
            command_param = int(command_param[-1])
            self.nn.setPredictWordsNumber(command_param)
            bot.send_message(chat_id=update.message.chat_id, text='☘️ Успешно ☘️')
        except Exception as e:
            bot.send_message(chat_id=update.message.chat_id, text='🔥 Ошибка 🔥')
            print(e)
            pass

    def beginWordsCommand(self, bot, update):
        try:
            command_param = update.message.text.rsplit(None, 1)
            command_param = int(command_param[-1])
            self.nn.setPredictBeginWordsNumber(command_param)
            bot.send_message(chat_id=update.message.chat_id, text='☘️ Успешно ☘️')
        except Exception as e:
            bot.send_message(chat_id=update.message.chat_id, text='🔥 Ошибка 🔥')
            print(e)
            pass

    def textMessage(self, bot, update):
        try:
            prediction = self.nn.predict(update.message.text)
            bot.send_message(chat_id=update.message.chat_id, text=prediction, parse_mode=tg.ParseMode.HTML)
        except Exception as e:
            print(e)
            bot.send_message(chat_id=update.message.chat_id, text='🔥 Ошибка 🔥')
            pass
    
    def __print(self, text):
        if self.sb != None:
            self.sb.print(text)
        print(text)