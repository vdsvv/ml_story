path_data='./data/[lermontov_azimov_vedmak]/'
path_models='./models/[lermontov_azimov_vedmak]/'
ps = dict(
        description='Модель model-v8.0[acc: 0.6000] генерирует абстрактные фразы.',
        default_model_href='https://drive.google.com/open?id=1UJ3P9W6aOWAsnb1LxZnu0Auc-kb-UY28',
        path_input_text_tokens=path_data + 'input_text_tokens.txt',
        path_target_text_tokens=path_data + 'target_text_tokens.txt',
        path_input_digit_tokens=path_data + 'input_digit_tokens.txt',
        path_target_digit_tokens=path_data + 'target_digit_tokens.txt',
        path_vocab=path_data + 'vocab.txt',
        path_models=path_models,
        default_model_name='model_sgnn-v8.0',
        default_model_ext='.hdf5',
        crop_lines=None,
        save_model_auto=True,
        save_model_auto_period=1,
        use_tensor_board=False,
        epochs=500,
        model_steps_num=30,
        skip_step=1,
        batch_size=100,
        validation_split=0.1,
        rnn_units=300,
        rnn_gpu_unit=False,
        rnn_bidirectional=False,
        rnn_layers=2,
        embeding_dim=128,
        stateful=False,
        dropout=0.6,
        pred_temperature=None,
        pred_words_num=100,
        pred_begin_words_num=None,
        use_service_bot=True,
        service_bot_token='',
        work_bot_token='',
        bot_need_proxy=False,
        bot_proxy_proto='socks5',
        bot_proxy_url='',#abracadabra.com:55655
        bot_proxy_user='itusiser',
        bot_proxy_pass='resipasutiss')

#ps['default_model_name'] = 'MODEL_STEP{}-SKIP{}-EDIM{}-LSTM{}'.format(
#                                      ps['model_steps_num'], 
#                                      ps['skip_step'],
#                                      ps['embeding_dim'],
#                                      ps['rnn_units'])
#if ps['stateful']:
#    ps['default_model_name'] += '-SFULL-BATCH{}'.format(ps['batch_size'])