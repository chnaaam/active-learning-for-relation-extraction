import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from re_config import get_configuration
from utils import initialize_directories, fix_torch_seed
from tokenizer import ReTokenizer
from data_loader import ReDataModule
from re_model_trainer import ReModelTrainer


def train(config_path, config_file):
    fix_torch_seed()

    config = get_configuration(
        config_path=config_path,
        config_file=config_file)

    initialize_directories(config=config)

    model_type = f"{config.model.embedding_layer}-{config.model.middle_layer}-{config.model.output_layer}"
    wandb_name = f"{model_type} batch:{config.parameters.batch_size} lr:{config.parameters.learning_rate} weight_decay: {config.parameters.weight_decay}"

    wandb_logger = WandbLogger(
        name=wandb_name,
        project="RE"
    )

    re_tokenizer = ReTokenizer(tokenizer_name=config.tokenizer.name)
    re_data_module = ReDataModule(
        config=config,
        model_type=model_type,
        tokenizer=re_tokenizer)

    re_model_trainer = ReModelTrainer(
        config=config,

        l2i=re_data_module.get_l2i(),
        i2l=re_data_module.get_i2l(),

        token_pad_id=re_data_module.get_token_pad_id())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=config.path.model,
        filename=f"re-{model_type}" + "-{val_acc:0.4f}-{val_loss:0.4f}",
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
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(re_model_trainer, re_data_module)

if __name__ == "__main__":
    train(
        config_path="./",
        config_file="re.cfg"
    )