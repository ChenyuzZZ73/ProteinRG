import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data.util import embed_labels
from utill.constants import amino_acid_alphabet
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Conv1D, LayerNormalization,Concatenate, Dense, BatchNormalization, Input, ReLU, LeakyReLU, Softmax, Flatten, Dot, Add, Layer, Lambda, Conv2DTranspose, Reshape, MultiHeadAttention
from tensorflow.keras import Sequential
from keras import Sequential
from keras.optimizers import Adam
from metrics import identity
from keras.layers import Layer, Input, Conv1D,Dense, BatchNormalization, Flatten
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import os, json
from data.util import tokenize, tokenize_labels, detokenize_sequences
from utill.fasta import save_as_fasta
from metrics.similarity import mmd
from metrics.conditional import mrr
from metrics.diversity import entropy,distance
import itertools
from collections import defaultdict
import math, time
sns.set_style("whitegrid")
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(tf.config.list_physical_devices('GPU：0'))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

layer = MultiHeadAttention(num_heads=2, key_dim=128)

def batch_dont_shuffle(data, batch_size):
    sequences, labels, representation = data
    return [[sequences[i:i+batch_size], labels[i:i+batch_size], representation[i:i+batch_size]] for i in range(0, len(sequences), batch_size)]

def batch_and_shuffle(data, batch_size):
    sequences, labels, representation = data
    transposed = list(zip(sequences, labels,representation))
    random.shuffle(transposed)
    shuffled = list(zip(*transposed))
    batched = batch_dont_shuffle(shuffled, batch_size)
    return batched
# Conv1DTranspose layer from: https://github.com/tensorflow/tensorflow/issues/6724 by Ryan Peach
class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)




class BaseTrainer():

    '''
    Base class for training a model. Takes care of the training loop,
    sequence generation and plotting. Subclass BaseTrainer and implement
    the indicated methods for your own model.
    '''

    def __init__(self,
            train_data = None,
            test_data = None,
            val_data = None,
            batch_size = None,
            path = '.',
            plot_every = 1,
            checkpoint_every = 100,
            pad_sequence = False,
            pad_labels = False,
            add_eos_token = False,
            short_val = False,
            config = None,
            ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.steps_per_epoch = 0

        pad_sequence_to = False if not pad_sequence else config['seq_length']

        if not train_data is None:
            self.tokenized_train = tokenize(train_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)
            self.steps_per_epoch = math.ceil(len(train_data)/batch_size)
            self.plot_interval = math.ceil(self.steps_per_epoch*plot_every)
            self.checkpoint_interval = int(self.steps_per_epoch*checkpoint_every)
        if not test_data is None:
            self.tokenized_test = tokenize(test_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)
        if not val_data is None:
            self.tokenized_val = tokenize(val_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)

        self.path = path if path.endswith('/') else path+'/'
        os.makedirs(self.path, exist_ok=True)
        with open(self.path+'config.json','w') as file:
            json.dump(config,file)
        self.step = 0
        self.metrics = []
        self.best = {'MMD':0, 'MRR':0, 'ratio':0}
        self.gen_length = config['seq_length'] if not short_val else 300

    def train(self, epochs, progress=False, restore=True):
        self.epochs = epochs
        if restore:
            self.restore()
        else:
            self.create_optimizer()
        last_epoch = int(self.step/self.steps_per_epoch)
        if restore:
            print('Restored model from epoch {}'.format(last_epoch))
        if progress:
            it = tqdm(range(last_epoch,epochs+1), total=epochs, initial=last_epoch)
        else:
            it = range(last_epoch,epochs+1)
        for epoch in it:
            self.train_epoch()
        self.store()

    def train_epoch(self):
        x = batch_and_shuffle(self.tokenized_train, self.batch_size)
        losses = defaultdict(lambda: [])
        for batch in batch_and_shuffle(self.tokenized_train, self.batch_size):
            loss = self.train_batch(batch)
            self.step += 1
            losses = {k: losses[k]+[v] for k,v in loss.items()}
            if self.step % self.plot_interval == 0:
                losses = {k:np.mean(v) for k,v in losses.items()}
                metrics = {'Step':self.step, **losses, **self.eval(seed=0)}
                self.metrics.append(metrics)
                self.plot()
                losses = defaultdict(lambda: [])
                if metrics['MRR']/metrics['MMD'] > self.best['ratio']:
                    #self.store(best=True)
                    self.best = {
                        'MMD': metrics['MMD'],
                        'MRR': metrics['MRR'],
                        'ratio': metrics['MRR']/metrics['MMD']
                    }
            #if self.step % self.checkpoint_interval == 0:
            #    self.store()

    def create_optimizer(self):
        '''
        Implement yourself. Should create an optimizer for the model.
        '''
        raise NotImplementedError

    def train_batch(self, batch):
        '''
        Implement yourself. Should train a single batch and return a dict of losses.
        '''
        raise NotImplementedError

    def predict_batch(self, batch, seed=None):
        '''
        Implement yourself. Should return a matrix of tokenized sequences.
        '''
        raise NotImplementedError

    def eval(self, seed=None):
        val_labels = self.val_data['labels'].tolist()
        val_seqs = self.val_data['sequence'].tolist()
        generated = []
        labels = []
        for batch in batch_dont_shuffle(self.tokenized_val, self.batch_size):
            seqs = self.predict_batch(batch, seed=seed)
            seqs = detokenize_sequences(seqs)
            generated.extend(seqs)
        #save_as_fasta(zip(generated,val_labels), self.path+'generated.fasta')
        if self.step % 1200 == 0:
            save_as_fasta(zip(generated, val_labels), self.path + 'generated'+ str(self.step)+'.fasta')

        metrics = {
            'MMD': mmd(generated, val_seqs),
            'MRR': mrr(generated, val_labels, val_seqs, val_labels, warning=False),
            'Entropy': entropy(generated, val_seqs),
            'Distance': distance(generated,val_seqs),
        }
        return metrics

    def plot(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv('metric.csv')
        plt.figure()
        colors = ['#f44336','#b23c17',"#82B0D2", "#FA7F6F",'#2196f3','#00bcd4','#ffc107','#9c27b0','#009688','#8bc34a']
        names = sorted(list(set(df.columns) - set(('MMD','MRR','Distance','Entropy','Step'))))
        for i, name in enumerate(names):
            ax = sns.lineplot(data=df, x='Step', y=name, legend=False, color=colors[i+4])
        plt.xlabel('Parameter Update', fontdict = {'size': 12})
        plt.ylabel('Loss', fontdict = {'size': 12})
        ax.grid(False)
        ax2 = ax.twinx()
        ax = sns.lineplot(data=df, x='Step', y='MMD', legend=False, ax=ax2, color=colors[0])
        ax = sns.lineplot(data=df, x='Step', y='MRR', legend=False, ax=ax2, color=colors[1])
        ax = sns.lineplot(data=df, x='Step', y='Distance', legend=False, ax=ax2, color=colors[2])
        ax = sns.lineplot(data=df, x='Step', y='Entropy', legend=False, ax=ax2, color=colors[3])
        plt.ylabel('MMD & MRR')
        lines = [Line2D([0], [0], color=colors[i], lw=2) for i, name in enumerate(['MMD','MRR','Distance','Entropy']+names)]
        ax.legend(lines, ['MMD','MRR','Distance','Entropy']+names, bbox_to_anchor=(0,0.98,1,0.2), loc="upper left", mode="expand", borderaxespad=0, ncol=4, frameon=False)
        plt.tight_layout()
        plt.savefig(self.path+'try.png', dpi=350)
        plt.close()

    def generate(self, labels=None, progress=False, return_df=False, seed=None):
        if labels is None:
            labels = self.test_data['labels'].tolist()
        #tokenized_labels = [tokenize_labels(l) for l in labels]
        tokenized_labels = [int(l) for l in labels]
        data = ([[]]*len(tokenized_labels),tokenized_labels)
        batched_data = batch_dont_shuffle(data, self.batch_size)
        it = tqdm(batched_data) if progress else batched_data
        generated = [self.predict_batch(batch, seed=seed) for batch in it]
        generated = np.array(list(itertools.chain(*generated)))
        generated = detok2enize_sequences(generated)
        if return_df:
            return pd.DataFrame({'sequence':generated, 'labels':labels})
        else:
            return generated

    def store(self, best=False):
        dir = 'Good/' if best else 'checkpoint/'
        os.makedirs(self.path+dir, exist_ok=True)
        props = {
                'best': self.best,
                'step': self.step
                }
        with open(self.path+dir+'props.json','w') as file:
            json.dump(props, file)
        with open(self.path+dir+'metrics.json','w') as file:
            json.dump(self.metrics, file)
        self.store_model(self.path+dir)

    def restore(self, path=None, best=False):
        path = self.path if path is None else path
        path = path if path.endswith('/') else path+'/'
        dir = 'Good/' if best else 'checkpoint/'
        if not os.path.exists(path+dir):
            raise 'Checkpoint for restoring the Trainer not found.'
        with open(path+dir+'props.json','r') as file:
            props = json.load(file)
        with open(path+dir+'metrics.json','r') as file:
            self.metrics = json.load(file)
        self.best = props['best']
        self.step = props['step']
        self.create_optimizer()
        self.restore_model(path+dir)

    def store_model(self, path):
        '''
        Implement yourself. Should save all model-related files and optimizer state.
        '''
        raise NotImplementedError

    def restore_model(self, path):
        '''
        Implement yourself. Should reload all model-related files and optimizer state.
        '''
        raise NotImplementedError


class ProteoGAN():
    '''
    Base class for ProteoGAN, which is based on a Wasserstein Generative Adversarial Network with Gradient Penalty (Gulrajani et al., 2017).
    '''
    def __init__(self, config):

        self.config = config
        self.seq_length = self.config['seq_length']
        self.seq_dim = self.config['seq_dim']
        self.label_dim = self.config['label_dim']
        self.z_dim = self.config['z_dim']
        self.strides = self.config['strides']
        self.kernel_size = self.config['kernel_size']
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.step = 0
        self.gen_loss = []
        self.gen_loss_buffer = []
        self.dis_loss = []
        self.dis_loss_buffer = []


    def build_generator(self):
        '''
        Builds the generator.
        '''

        z_input = Input(shape=(self.z_dim,))
        #c_input = Input(shape=(11,))
        c_input = Input(shape=(480,))
        x = Concatenate(axis=1)([z_input,c_input])
        x = Dense(units=self.seq_length * self.seq_dim, activation='relu')(x)
        x = BatchNormalization()(x)
        L = int(self.seq_length / (self.strides ** 2))
        f = int(self.seq_length * self.seq_dim / L)
        x = Reshape(target_shape=(L, f))(x)
        L = int(self.seq_length / (self.strides ** 1))
        f = int(self.seq_length * self.seq_dim / L)
        x = Conv1DTranspose(filters=f, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv1DTranspose(filters=self.seq_dim, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = ReLU()(x)
        x = Softmax(axis=-1)(x)
        output = x
        return KerasModel([z_input, c_input], [output])

    def build_discriminator(self):
        '''
        Builds the discriminator.
        '''

        projections = []
        def project(x):
            x = Flatten()(x)
            x = Dense(self.label_dim)(x)
            c = Dense(self.label_dim)(c_input)
            #c = c_input
            dot = Dot(axes=1)([x,c])
            x = Dense(1)(x)
            output = Add()([dot,x])
            projections.append(output)

        x_input = Input(shape=(self.seq_length, self.seq_dim))
        #c_input = Input(shape=(11,))
        c_input = Input(shape=(480,))
        L = int(self.seq_length/(self.strides**1))
        f = int(self.seq_length*self.seq_dim/L)
        x = Conv1D(filters=f, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x_input)
        x = LeakyReLU(alpha=0.2)(x)
        project(x)
        x = Conv1D(filters=256, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x, weights = MultiHeadAttention(num_heads=2, key_dim=128)(x, x, return_attention_scores=True)
        project(x)
        x = Flatten()(x)
        output_source = Add()(projections)
        output_labels = Dense(units=self.label_dim, activation='sigmoid')(x)
        return KerasModel([x_input, c_input],[output_source,output_labels])


class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = kwargs['config']
        self.seq_length = self.config['seq_length']
        self.seq_dim = self.config['seq_dim']
        self.label_dim = self.config['label_dim']
        self.z_dim = self.config['z_dim']
        self.strides = self.config['strides']
        self.kernel_size = self.config['kernel_size']
        self.ac_weight = self.config['ac_weight']

        self.model = ProteoGAN(self.config)

    def train_batch(self, batch):
        '''
        Train one batch on generator and discriminator.
        '''
        seq, labels, re = batch
        re = np.array(re)
        labels = np.array([embed_labels(tokens) for tokens in labels])
        z = self.sample_z(labels.shape[0])
        dis_losses = self.discriminator_train_step([seq, labels,re])
        gen_losses = self.generator_train_step([z, labels,re])
        losses = {**gen_losses, **dis_losses}
        losses = {k:float(v.numpy()) for k,v in losses.items()}
        return losses

    def create_optimizer(self):
        '''
        Creates the optimizer.
        '''
        self.generator_optimizer = Adam(4e-5, beta_1=0.0, beta_2=0.9)
        self.discriminator_optimizer = Adam(4e-5, beta_1=0.0, beta_2=0.9)

    def predict_batch(self, batch, seed=None):
        '''
        Generate a sample with a given <batch> of latent variables and conditioning labels.
        '''
        seq, labels,re = batch
        labels = np.array([embed_labels(tokens) for tokens in labels])
        re = np.array(re)
        z = self.sample_z(labels.shape[0], seed=seed)
        output = self._predict_batch([z, re])
        return np.array(tf.argmax(output, axis=-1)).astype(int)

    @tf.function
    def _predict_batch(self, batch):
        return self.model.generator(batch, training=False)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, real_label_output, fake_label_output, real_labels, gradient, L=10):
        '''
        WGAN-GP loss.
        '''
        norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)
        gradient_penalty = (norm - 1.)**2
        w_loss = K.mean(fake_output) - K.mean(real_output) + L * K.mean(gradient_penalty)
        ac_loss = self.ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, real_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    @tf.function
    def generator_loss(self, fake_output, fake_label_output, real_labels):
        '''
        WGAN-GP loss.
        '''
        w_loss = K.mean(-fake_output)
        ac_loss = self.ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, fake_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    @tf.function
    def sample_z(self, batch_size, seed=None):
        '''
        Generates a latent noise vector of <batch_size> instances. Optionally with fixed <seed> for reproducibility.
        '''
        return tf.random.normal((batch_size, self.z_dim),seed=seed)

    #@tf.function
    def classify(self, batch, labels):
        '''
        Classifiy a <batch> of sequences and labels with the auxiliary classifier.
        '''
        output, label_output = self.model.discriminator([batch,labels], training=False)
        return label_output

    @tf.function
    def generator_train_step(self, batch):
        '''
        A generator train step. Returns losses.
        '''
        c = batch[1]
        data_size = c.shape[0]
        re = batch[2]
        z = self.sample_z(data_size)
        with tf.GradientTape() as gen_tape:
            generated_data = self.model.generator([z,re], training=True)
            fake_output, fake_label_output = self.model.discriminator([generated_data,re], training=True)
            gen_loss, w_loss, ac_loss  = self.generator_loss(fake_output, fake_label_output, c)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator.trainable_variables))
        return {'Total Loss (G)': gen_loss, 'W. Loss (G)': w_loss, 'AC Loss (G)': ac_loss}

    @tf.function
    def discriminator_train_step(self, batch):
        '''
        A discriminator train step. Returns losses.
        '''
        c = batch[1]
        re = batch[2]
        data = tf.one_hot(tf.cast(batch[0], tf.int32), depth=self.seq_dim, axis=-1, dtype=tf.dtypes.float32)
        data_size = data.shape[0]
        e_shape = (data_size,)
        for i in data.shape[1:]:
            e_shape = e_shape + (1,)
        z = self.sample_z(data_size)
        with tf.GradientTape() as disc_tape:
            generated_data = self.model.generator([z,re], training=True)
            real_output, real_label_output = self.model.discriminator([data,re], training=True)
            fake_output, fake_label_output = self.model.discriminator([generated_data,re], training=True)
            epsilon = K.random_uniform(e_shape, dtype=tf.dtypes.float32)
            random_weighted_average = (epsilon * data) + ((1 - epsilon) * generated_data)
            # calculate gradient for penalty
            with tf.GradientTape() as norm_tape:
                norm_tape.watch(random_weighted_average)
                average_output = self.model.discriminator([random_weighted_average,re], training=True)
            gradient = norm_tape.gradient(average_output, random_weighted_average)
            disc_loss, w_loss, ac_loss = self.discriminator_loss(real_output, fake_output, real_label_output, fake_label_output, c, gradient, L=10)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.discriminator.trainable_variables))
        return {'Total Loss (D)': disc_loss, 'W. Loss (D)': w_loss, 'AC Loss (D)': ac_loss}

    def store_model(self, path):
        '''
        Save a model checkpoint.
        '''
        self.model.generator.save_weights(path+'gen')
        self.model.discriminator.save_weights(path+'dis')

    def restore_model(self, path):
        '''
        Restore a model checkpoint.
        '''
        self.model.generator.load_weights(path+'gen')
        self.model.discriminator.load_weights(path+'dis')
