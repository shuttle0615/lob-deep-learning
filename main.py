from loggers import logger
from optimizers import train
from simulator import evaluate

if __name__ == '__main__':
    '''
    model_id = logger.generate_id('deeplob-lighten')
    train.train(
        model_id=model_id, dataset_type = 'krx', normalization= 'Zscore', lighten= True,
        T= 100, k= 100, stock= ['KS200', 'KQ150'], train_test_ratio = 0.7)
    evaluate.test(
        model_id=model_id, dataset_type='krx', normalization='Zscore', lighten=True,
        T=100, k=100, stock=['KS200', 'KQ150'], train_test_ratio=0.7)
    '''
    model_id = 'deeplob-lighten_2022-11-25_12:59:09'
    evaluate.test(model_id=model_id)
