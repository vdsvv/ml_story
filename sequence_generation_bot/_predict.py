
from sequence_generator_nn import SequenceGeneratorNN 
from params.lermontov_azimov_vedmak_2 import ps
from service_bot import ServiceBot

def predict():
    print('👉 checkPredictions')
    ps['model_steps_num'] = None
    ps['batch_size'] = 1
    sbot = ServiceBot(ps) if ps['use_service_bot'] else None
    nn = SequenceGeneratorNN(ps, sbot)
    nn.loadVocab()
    nn.createModel()
    nn_weights_path = ps['path_models'] + ps['default_model_name'] + ps['default_model_ext']
    nn.loadWeights(nn_weights_path)
    res = nn.predict('Здесь', 20)
    res = nn.predict('Здесь, в мутном свете остроконечных', 10)
    res = nn.predict('грибы с тонкими ножками:', 20)
    res = nn.predict('какое существовало во', 30)
    res = nn.predict('Начиная рассказ, рассказчик', 40)
    res = nn.predict('Бууу', 10)
    res = nn.predict('Это Бууу', 10)
    res = nn.predict('Бууу это', 10)
    res = nn.predict('Это', 10)
    res = nn.predict('чтобы никого важного', 15)
    res = nn.predict('За столом делалось все веселее.  Компания  Лютика', 20)
    res = nn.predict('Мышовур в обществе', 25)
    res = nn.predict('Ноги  Рагнара,', 30)
    res = nn.predict('- Повеселились-то хорошо', 40)
    res = nn.predict('Потом началось  повальное', 45)
    res = nn.predict('- Я рада', 50)
    res = nn.predict('В западном крыле', 55)
    res = nn.predict('Йеннифэр  отняла  руку', 60)
    print('👈 checkPredictions')

predict()