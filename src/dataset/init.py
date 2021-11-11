import logging

from .graph import Graph
from .feeder import Preprocess_Feeder




__feeder = {
    'kinetics': Preprocess_Feeder,
    'animal-skeleton':Preprocess_Feeder,
}

__shape = {
    'kinetics': [3,3,300,18,2],
    'animal-skeleton':[3,3,40,18,1]
}

__class = {
    'kinetics': 400,
    'animal-skeleton':5
}

def create(debug, dataset, path, preprocess=False, **kwargs):
    print(dataset)
    if dataset not in __class.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()
    graph = Graph(dataset)
    feeder_name = 'ntu-preprocess' if 'ntu' in dataset and preprocess else dataset
    kwargs.update({
        'path': '{}/{}'.format(path, dataset.replace('-', '/')),
        'data_shape': __shape[feeder_name],
        'connect_joint': graph.connect_joint,
        'debug': debug,
    })

    feeders = {
        'train': __feeder[feeder_name]('train', **kwargs),
        'eval' : __feeder[feeder_name]('val', **kwargs),
    }
    return feeders, __shape[feeder_name], __class[dataset], graph.A, graph.parts
