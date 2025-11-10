import sys, os, time
from data.util import load
from metrics.similarity import mmd
from metrics.conditional import mrr

PATH = os.path.dirname(os.path.realpath(__file__))

model = 'proteogan'
split = 3

assert model in ['proteogan']


train = load('train',3)
test = load('test', 3)
val = load('val', 3)

if model == 'proteogan':
    from models.gan import Trainer
    from models.config import proteogan_config as model_config

trainer = Trainer(
    batch_size = model_config['batch_size'],
    plot_every = 1,
    checkpoint_every = 10,
    train_data = train,
    test_data = test,
    val_data = val,
    path = '{}/ssplit_{}/{}'.format(PATH,split,model),
    config = model_config,
    pad_sequence = model_config['pad_sequence'],
    pad_labels = model_config['pad_labels'],
    short_val = model == 'transformer',
)

trainer.train(epochs=1000, progress=True, restore=False)
