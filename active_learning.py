import os
import torch

from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from re_config import get_configuration
from utils import initialize_directories, fix_torch_seed
from tokenizer import ReTokenizer
from data_loader import ReDataModule
from data_loader import ReDataManager
from re_model_trainer import ReModelTrainer
from prediction_model import PredictionModel

from active_learning_methods import least_confidence

def active_learning(config_path, config_file):
    fix_torch_seed()

    config = get_configuration(
        config_path=config_path,
        config_file=config_file)

    initialize_directories(config=config)

    model_type = f"{config.model.embedding_layer}-{config.model.middle_layer}-{config.model.output_layer}"

    re_tokenizer = ReTokenizer(tokenizer_name=config.tokenizer.name)

    re_data_manager = ReDataManager(
        config=config,
        model_type=model_type,
        tokenizer=re_tokenizer)

    for i in range(1, 11):
        print("Current data size for training : ", len(re_data_manager.seed_idxes))
        i = i * 10

        re_data_manager.set_data_module(re_data_manager.seed_idxes)
        train(config, model_type + f"-{i}", re_data_manager)
        convert_ckpt2weight(config, model_type + f"-{i}", re_data_manager.data_module)
        prob_list = predict(config, model_type + f"-{i}", re_data_manager.get_unlabeled_data())

        if config.active_learning.method == "least_confidence":
            selected_idx_list = least_confidence(prob_list=prob_list, size=re_data_manager.data_interval)

            re_data_manager.set_idx_list(idx_list=selected_idx_list)

        else:
            raise NotImplementedError()


def train(config, model_type, re_data_manager):
    re_model_trainer = ReModelTrainer(
        config=config,

        l2i=re_data_manager.data_module.get_l2i(),
        i2l=re_data_manager.data_module.get_i2l(),

        token_pad_id=re_data_manager.data_module.get_token_pad_id())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=config.path.model,
        filename=f"re-{model_type}" + "-{epoch:02d}-{val_acc:0.4f}-{val_loss:0.4f}",
        save_top_k=1,
        mode="max")

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0.01,
        patience=3,
        verbose=False,
        mode="max")

    # Train
    trainer = pl.Trainer(
        max_epochs=config.parameters.epoch,
        gpus=2,
        accelerator="dp",
        precision=16,
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(re_model_trainer, re_data_manager.data_module)


def convert_ckpt2weight(config, model_type, re_data_module):
    model_list = os.listdir(config.path.model)
    model_fn = ""
    for model_name in model_list:
        if model_type in model_name:
            model_fn = model_name
            break

    re_model_trainer = ReModelTrainer(
        config=config,

        l2i=re_data_module.get_l2i(),
        i2l=re_data_module.get_i2l(),

        token_pad_id=re_data_module.get_token_pad_id())

    path = os.path.join(config.path.model, model_fn)
    check_point = torch.load(path)
    re_model_trainer.load_state_dict(check_point["state_dict"])
    torch.save(
        re_model_trainer.model.state_dict(),
        os.path.join(config.path.hub, f"{model_type}.mdl")
    )

def predict(config, model_type, data_list):
    model = PredictionModel(config, model_type)

    prob_list = []
    for idx, data in tqdm(data_list):
        sentence, subj_start_idx, subj_end_idx, subj_label, obj_start_idx, obj_end_idx, obj_label, _ = data
        prob = model.predict(
            sentence=sentence,
            subj_start_idx=subj_start_idx,
            subj_end_idx=subj_end_idx,
            subj_label=subj_label,
            obj_start_idx=obj_start_idx,
            obj_end_idx=obj_end_idx,
            obj_label=obj_label,
            return_method="prob"
        )

        prob_list.append((idx, prob[0]))

    return prob_list


if __name__ == "__main__":
    active_learning(
        config_path="./",
        config_file="re.cfg"
    )