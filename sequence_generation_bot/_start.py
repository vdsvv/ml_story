from sequence_generator_nn import SequenceGeneratorNN
from service_bot import ServiceBot
from generate_text_bot import GenerateTextBot
from params.lermontov_azimov_vedmak_2 import ps

def startBot():
    try:
        sbot = ServiceBot(ps) if ps['use_service_bot'] else None
        print('🔔 START GEN TEXT BOT') 
        if sbot != None: sbot.print('🔔  START GEN TEXT BOT') 
        ps['model_steps_num'] = None
        ps['batch_size'] = 1
        nn = SequenceGeneratorNN(ps, sbot)
        nn.loadVocab()
        nn.createModel()
        nn_weights_path = ps['path_models'] + ps['default_model_name'] + ps['default_model_ext']
        nn.loadWeights(nn_weights_path)
        nn.model.trainable = False
        nn.predict('.', 3)
        bot = GenerateTextBot(nn, ps, sbot)
        bot.start()
        print('🔕 STOP GEN TEXT BOT')
        if sbot != None: sbot.print('🔕 STOP GEN TEXT BOT')
    except Exception as e:
        print('🔥 ERROR GEN TEXT BOT: ' + str(e))
        if sbot != None: sbot.print('🔥 ERROR GEN TEXT BOT: ' + str(e))


def checkBot():
    try:
        sbot = ServiceBot(ps) if ps['use_service_bot'] else None
        print('🔔 START GEN TEXT BOT') 
        if sbot != None: sbot.print('🔔  START GEN TEXT BOT') 
        bot = GenerateTextBot(None, ps, sbot)
        bot.start()
        print('🔕 STOP GEN TEXT BOT')
        if sbot != None: sbot.print('🔕 STOP GEN TEXT BOT')
    except Exception as e:
        print('🔥 ERROR GEN TEXT BOT: ' + str(e))
        if sbot != None: sbot.print('🔥 ERROR GEN TEXT BOT: ' + str(e))

startBot()
#checkBot()