import os

# change directory to the dataset where our
# custom scripts are found
os.chdir("/kaggle/input/textgen-1")
#os.chdir("../input/")

# read in custom modules 
from sequence_generator_nn import SequenceGeneratorNN
from service_bot import ServiceBot
#from generate_text_bot import GenerateTextBot

#from params.lermontov_azimov_vedmak_2 import ps as ps_1

# reset our working directory
os.chdir("/kaggle/working/")

path_data='/kaggle/input/textgen-1/'
#path_data='/kaggle/input/'
#path_models='/kaggle/input/textgen-1/'
path_models=''
ps_1 = dict(
        description='Экспериментальная модель',
        default_model_href='https://drive.google.com/open?id=1UJ3P9W6aOWAsnb1LxZnu0Auc-kb-UY28',
        path_input_text_tokens=path_data + 'input_text_tokens.txt',
        path_target_text_tokens=path_data + 'target_text_tokens.txt',
        path_input_digit_tokens=path_data + 'input_digit_tokens.txt',
        path_target_digit_tokens=path_data + 'target_digit_tokens.txt',
        path_vocab=path_data + 'vocab.txt',
        path_models=path_models,
        default_model_name='model-v12.0',
        default_model_ext='.hdf5',
        crop_lines=None,
        save_model_auto=True,
        save_model_auto_period=1,
        use_tensor_board=False,
        epochs=1,
        model_steps_num=30,
        skip_step=1,
        batch_size=20,
        validation_split=0.1,
        rnn_units=300,
        rnn_bidirectional=False,
        rnn_layers=2,
        rnn_gpu_unit=True,
        embeding_dim=128,
        stateful=False,
        dropout=0.6,
        pred_temperature=None,
        pred_words_num=100,
        pred_begin_words_num=None,
        use_service_bot=True,
        service_bot_token='',
        work_bot_token='',
        bot_need_proxy=True,
        bot_proxy_proto='socks5',
        bot_proxy_url='',
        bot_proxy_user='itusiser',
        bot_proxy_pass='resipasutiss')

def trainNN():
    sbot = ServiceBot(ps_1) if ps_1['use_service_bot'] else None
    print('🔔 START TRAIN') 
    if sbot != None: sbot.print('🔔  START TRAIN') 
    nn = SequenceGeneratorNN(ps_1, sbot)
    nn.generateData()
    nn.createModel()
    #nn.loadWeights('../input/textgen-1/model-v8.0.hdf5')
    nn.train()
    #nn.saveWeights()
    print('🔕 STOP TRAIN')
    if sbot != None: sbot.print('🔕 STOP TRAIN')
trainNN()

#def startNN():
#    bot = GenerateTextBot(ps_1)
#    bot.start()
#startNN()