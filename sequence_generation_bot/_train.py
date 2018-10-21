# tensorboard --logdir ./graph

from sequence_generator_nn import SequenceGeneratorNN 
from params.lermontov_azimov_vedmak_2 import ps
from service_bot import ServiceBot

def trainNN():
    sbot = ServiceBot(ps) if ps['use_service_bot'] else None
    print('🔔 START TRAIN') 
    if sbot != None: sbot.print('🔔  START TRAIN') 
    nn = SequenceGeneratorNN(ps, sbot)
    nn.generateData()
    nn.createModel()
    nn.loadWeights()
    nn.train()
    #nn.saveWeights()
    print('🔕 STOP TRAIN')
    if sbot != None: sbot.print('🔕 STOP TRAIN')

trainNN()