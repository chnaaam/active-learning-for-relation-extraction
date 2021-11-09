import pytorch_lightning as pl

from re_config import get_configuration
from utils import initialize_directories, fix_torch_seed
from tokenizer import ReTokenizer
from data_loader import ReDataModule
from re_model_trainer import ReModelTrainer



def predict(config_path, config_file):

    config = get_configuration(
        config_path=config_path,
        config_file=config_file)

    initialize_directories(config=config)

    model_type = f"{config.model.embedding_layer}-{config.model.middle_layer}-{config.model.output_layer}"

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

    # Train
    trainer = pl.Trainer(
        max_epochs=config.parameters.epoch,
        gpus=2,
        accelerator="dp",
        precision=16,
        callbacks=[checkpoint_callback])

    trainer.fit(re_model_trainer, re_data_module)

if __name__ == "__main__":
    predict(
        config_path="./",
        config_file="re.cfg"
    )