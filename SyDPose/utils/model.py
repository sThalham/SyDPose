
def freeze(model):

    for layer in model.layers:
        layer.trainable = False
    return model
