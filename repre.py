import esm
import torch
from utill.constants import amino_acid_alphabet
import torch
import seaborn as sns
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import esm
from tqdm import tqdm


from transformers import T5Tokenizer, AutoTokenizer,T5EncoderModel, AutoModel
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
model = model.to(device)
print(model)



for name,param in model.named_parameters():
    param.requires_grad =False



with open('data/labels_lyso.txt') as file:
    labels = file.read().split()

seq_tokens = list('-'+amino_acid_alphabet)
label_tokens = labels
all_tokens = np.array(seq_tokens + label_tokens)

aa_map = {k:v for v,k in enumerate(seq_tokens)}
print(aa_map)
seq_token_map = {v:k for k,v in aa_map.items()}
label_map = {k:v for v,k in enumerate(label_tokens)}
print(label_map)




def tokenize_labels(labels, pad=False):
    labels = labels.split()
    if pad:
        all_labels = list(filter(lambda x: x != '<PAD>', label_map.keys()))
        labels = [k if k in labels else '<PAD>' for k in all_labels] #labels + ['<PAD>']*(50-len(labels))
    return np.array([label_map[label] for label in labels])



from goatools.obo_parser import GODag as _GoDag
from goatools.godag.go_tasks import get_go2ancestors

class GoDag():
    '''
    GO DAG class to represent the ontology. Contains helper functions to get all parents or leaf nodes.
    '''
    def __init__(self, path='data/godag.obo'):
        self.GODAG = _GoDag(path, prt=None)

    def get(self, term):
        return self.GODAG[term]

    def get_go_lineage_of(self, terms):
        g = [self.GODAG[i] for i in terms]
        g = get_go2ancestors(g, False)
        gos = []
        for key in g:
            gos.append(key)
            gos.extend(g[key])
        return list(set(gos))

    def get_leaf_nodes(self, terms):
        gos = set(terms)
        if len(gos) > 1:
            leaves = []
            for go in gos:
                childs = set([i.id for i in self.get(go).children])
                inter = gos.intersection(childs)
                if len(inter) == 0:
                    leaves.append(go)
            if len(leaves) > 1:
                gos = set(leaves)
        return list(gos)

def one_hot_label(label):
    a = np.zeros(61,dtype='float32')
    for tokens in label:
        a[tokens] = 1
    return a




import time
import torch
from torch import nn
import args
import os
import torch.nn.functional as F
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
import matplotlib.pyplot as plt
import itertools
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from torch.optim import Adam




class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()

        self.config = config
        # self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.min_loss = {
            'Tox' + "min_valid_loss": torch.finfo(torch.float32).max,
            'Tox' + "min_epoch": 0,
        }

        # Word embeddings layer
        # input embedding stem

        self.pos_emb = None
        self.drop = nn.Dropout(0.5)
        ## transformer
        self.blocks = model

        self.train_config = config
        # if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  # nn.ModuleList()
        self.loss = torch.nn.CrossEntropyLoss()

        self.net = self.Net(
            640, 61, dropout=0.5,
        )

    class Net(nn.Module):
        def __init__(self, smiles_embed_dim, num_classes,  dropout=0.5):
            super().__init__()
            self.desc_skip_connection = True
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc = nn.Linear(640, smiles_embed_dim)

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, 512)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(512, num_classes)  # classif

        def forward(self, smiles_emb):
            smiles_emb = self.fc(smiles_emb)
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            #if self.desc_skip_connection is True:
                #z = self.final(z + x_out)
            #else:
            z = self.final(z)

            # z = self.layers(smiles_emb)
            return z


    def get_loss(self, smiles_emb, measures):
        z_pred = self.net.forward(smiles_emb)
        measures = measures
        # print('z_pred:', z_pred.shape)
        # print('measures:', measures.shape)
        return self.loss(z_pred, measures), z_pred, measures

    def on_save_checkpoint(self, checkpoint):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state'] = torch.get_rng_state()
        out_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state'] = np.random.get_state()
        if random:
            out_dict['python_state'] = random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        # load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key == 'torch_state':
                torch.set_rng_state(value)
            elif key == 'cuda_state':
                torch.cuda.set_rng_state(value)
            elif key == 'numpy_state':
                np.random.set_state(value)
            elif key == 'python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        # optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        optimizer = Adam(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        seq = batch[1]
        targets = batch[0]
        loss = 0
        loss_tmp = 0
        token_embeddings = self.tokenizer(seq, padding='max_length', max_length=2048, return_tensors='pt')
        token_embeddings = token_embeddings.to(device)
        x = self.blocks(**token_embeddings)
        loss_input = x.last_hidden_state[:,0,:].squeeze()
        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('train_loss', loss, on_step=True)

        logs = {"train_loss": loss}

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        seq = val_batch[1]
        targets = val_batch[0]
        loss = 0
        loss_tmp = 0
        token_embeddings = self.tokenizer(seq, padding='max_length', max_length=2048, return_tensors='pt')
        token_embeddings = token_embeddings.to(device)
        x = self.blocks(**token_embeddings)
        loss_input = x.last_hidden_state[:, 0, :].squeeze()
        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('test_loss', loss, on_step=True)
        return {
            "val_loss": loss,
            "pred": pred.detach(),
            "actual": actual.detach(),
            "dataset_idx": dataset_idx,
        }

    def validation_epoch_end(self, outputs):
        # results_by_dataset = self.split_results_by_dataset(outputs)
        tensorboard_logs = {}
        for dataset_idx, batch_outputs in enumerate(outputs):
            dataset = ['valid', 'test'][dataset_idx]
            # print("x_val_loss:", batch_outputs[0]["val_loss"])
            avg_loss = torch.stack([x["val_loss"] for x in batch_outputs]).mean()
            preds = torch.cat([x["pred"] for x in batch_outputs])
            actuals = torch.cat([x["actual"] for x in batch_outputs])
            val_loss = self.loss(preds, actuals)

            tensorboard_logs.update(
                {
                    # dataset + "_avg_val_loss": avg_loss,
                    'Tox' + "_" + dataset + "_loss": val_loss,
                }
            )

        if (
                tensorboard_logs['Tox' + "_valid_loss"]
                < self.min_loss['Tox' + "min_valid_loss"]
        ):
            self.min_loss['Tox' + "min_valid_loss"] = tensorboard_logs[
                'Tox' + "_valid_loss"
                ]
            self.min_loss['Tox' + "min_test_loss"] = tensorboard_logs[
                'Tox' + "_test_loss"
                ]
            self.min_loss['Tox' + "min_epoch"] = self.current_epoch


        tensorboard_logs['Tox' + "_min_valid_loss"] = self.min_loss[
            'Tox' + "min_valid_loss"
            ]
        tensorboard_logs['Tox' + "_min_test_loss"] = self.min_loss[
            'Tox' + "min_test_loss"
            ]

        self.logger.log_metrics(tensorboard_logs, self.global_step)

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k])

        print("Validation: Current Epoch", self.current_epoch)
        append_to_file(
            os.path.join('checkpoints_/measure/results/result.csv'),
            f"{'Tox'}, {self.current_epoch},"
            + f"{tensorboard_logs['Tox' + '_valid_loss']},"
            + f"{tensorboard_logs['Tox' + '_test_loss']},"
            + f"{self.min_loss['Tox' + 'min_epoch']}",
        )

        return {"avg_val_loss": avg_loss}



def get_dataset(data_root, filename, dataset_len):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df)
    return dataset


class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        super().__init__()
        df = df
        self.label = [tokenize_labels(seq) for seq in df['labels'].tolist()]
        self.seq = df['sequence']

    def __len__(self):
        return len(self.seq)  # 数据集长度

    def __getitem__(self, index):
        labels= self.label[index]
        labels = one_hot_label(labels)
        seq = self.seq[index]
        return labels, seq



class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        # self.hparams = Namespace(n_head=12, fold=0, n_layer=12, d_dropout=0.1, n_embd=768, fc_h=512, n_batch=512, from_scratch=False, checkpoint_every=1000, lr_start=3e-05, lr_multiplier=1, n_jobs=1, device='cuda', seed=12345, seed_path='', num_feats=32, max_epochs=500, mode='avg', train_dataset_length=None, eval_dataset_length=None, desc_skip_connection=False, num_workers=8, dropout=0.1, dims=[768, 768, 768, 1], smiles_embedding='/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/embeddings/protein/ba_embeddings_tanh_512_2986138_2.pt', aug=None, num_classes=2, dataset_name='hiv', measure_name='HIV_active', checkpoints_folder="'./checkpoints_hiv'", checkpoint_root=None, data_root='../data/data/hiv', batch_size=32)
        super(PropertyPredictionDataModule, self).__init__()
        self.smiles_emb_size = 640
        self.dataset_name = 'split_2'

    def get_split_dataset_filename(dataset_name, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "val"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            './data/split_2/',
            train_filename,
            None
        )

        val_ds = get_dataset(
            './data/split_2/',
            valid_filename,
            None
        )

        test_ds = get_dataset(
            './data/split_2/',
            test_filename,
            None
        )

        self.train_ds = train_ds
        self.val_ds = [val_ds] + [test_ds]

        # print(
        #     f"Train dataset size: {len(self.train_ds)}, val: {len(self.val_ds1), len(self.val_ds2)}, test: {len(self.test_ds)}"
        # )

    def collate(self, batch):
        return (torch.FloatTensor([smile[0] for smile in batch]), [smile[1] for smile in batch])

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=32,
                num_workers=8,
                shuffle=False,
                collate_fn=self.collate,
            )
            for ds in self.val_ds
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=32,
            num_workers=8,
            shuffle=True,
            collate_fn=self.collate,
        )


class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
                 prefix="N-Step-Checkpoint",
                 use_modelcheckpoint_filename=False,
                 ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
                # filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def main():
    margs = args.parse_args()
    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")


    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))

    print(margs)
    datamodule = PropertyPredictionDataModule(margs)
    margs.dataset_names = "valid test".split()
    margs.run_name = run_name

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.measure_name)
    margs.checkpoint_root = checkpoint_root
    margs.run_id = np.random.randint(30000)
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models_" + str(margs.run_id))
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, dirpath=checkpoint_dir, filename='checkpoint',
                                                       verbose=True)

    print(margs)

    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        # version=run_name,
        name="lightning_logs",
        default_hp_metric=False,
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    # seed.seed_everything(margs.seed)
    model = LightningModule(margs, tokenizer)
    print(model)

    # if margs.seed_path == '':
    #    print("# training from scratch")
    #    model = LightningModule(margs, tokenizer)
    # else:
    # print("# loaded pre-trained model from {args.seed_path}")
    # model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))

    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from_checkpoint = None
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from scratch")

    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        logger=logger,
        gpus=[3],
        # resume_from_checkpoint=resume_from_checkpoint,
        # checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
    )

    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()
