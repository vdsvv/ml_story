from sentiment_nn import SentimentNN 

def trainNN():
    print('>>trainNN')
    nn = SentimentNN()
    nn.generateRusTweetData()
    nn.createModel()
    nn.loadWeights('.\\model_weights_rus_tweet_1.h5py')
    nn.train(epochs = 10, batch_size = 100)
    nn.saveWeights('.\\model_weights_rus_tweet_2.h5py')
    nn.evaluate()
    print('<<trainNN')

def checkPredictions():
    print('checkPredictions')
    nn = SentimentNN()
    nn.createModel()
    nn.loadWeights('.\\model_weights_rus_tweet_1.h5py')
    
    res = nn.predict("""
    Работа не волк, 
    в лес не убежит.
    """)
    res = nn.predict('Это очень хорошо')
    res = nn.predict('Я не люблю тебя')
    res = nn.predict('Обожаю собак и кошек')
    res = nn.predict('Я простыл')
    res = nn.predict('Отравился сегодня и умер')
    res = nn.predict('Кошка лучший друг человека')
    res = nn.predict('Мне не нравится работать')
    res = nn.predict('Прекрасная погода сегодня утром')
    print('<<checkPredictions')

#trainNN()
checkPredictions()