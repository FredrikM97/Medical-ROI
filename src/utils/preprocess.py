import numpy as np

def normalize(x):
    return (x - x.min())/(x.max()-x.min())

def batchisize_to_5D(x):
    return x.expand((*[1]*(5-len(x.shape)),*[-1]*len(x.shape)))

def filename2labels(images:list, classes:dict, delimiter:str):
    assert delimiter in images[0], "The defined delimiter for ClassWeights could not be found in image input!"
    return np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])
