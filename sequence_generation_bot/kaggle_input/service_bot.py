# https://proglib.io/p/telegram-bot/
import requests

class ServiceBot:

    def __init__(self, ps):
        self.api_url = "https://api.telegram.org/bot{}/".format(ps['service_bot_token'])
        if ps['bot_need_proxy']:
            self.proxi_addr = '{}:{}:{}@{}'.format(ps['bot_proxy_proto'], ps['bot_proxy_user'], ps['bot_proxy_pass'], ps['bot_proxy_url'])
            #self.proxi_addr = 'socks5:itusiser:resipasutiss@iokjh.com:55655'
            self.proxies = {'http': self.proxi_addr, 'https': self.proxi_addr}
        else:
            self.proxies = None
        self.headers={}
        self.start()

    def start(self):
        try:
            last_update = self.get_updates(None, 30)
            self.last_update_id = last_update['update_id']
            #self.last_chat_text = last_update['message']['text']
            self.last_chat_id = last_update['message']['chat']['id']
            self.last_chat_name = last_update['message']['chat']['first_name']
        except Exception as e:
            print('🔥 ServiceBot Error: ' + str(e))
    
    def get_updates(self, offset=None, timeout=30):
            method = 'getUpdates'
            params = {'timeout': timeout, 'offset': offset}
            resp = requests.get(self.api_url + method, params, headers=self.headers, proxies=self.proxies)
            result_json = resp.json()['result']
            res = result_json[-1]
            return res
        
    def send_message(self, text):
        try:
            params = {'chat_id': self.last_chat_id, 'text': text, 'parse_mode': 'HTML'}
            method = 'sendMessage'
            resp = requests.post(self.api_url + method, params, headers=self.headers, proxies=self.proxies)
            return resp
        except Exception as e:
            print('🔥 ServiceBot Error: ' + str(e))
    
    def print(self, text):
        self.send_message('<b>{}</b>'.format(text))
    
    def print_params(self, ps):
        text = '<b>🍀Parameters:</b>\n\n'
        for par in ps.items():
            text += '<b>{}:</b> <code>{}</code>\n'.format(par[0], par[1])
        self.send_message(text)