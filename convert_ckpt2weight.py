import os
import torch

from re_config import get_configuration
from utils import initialize_directories, fix_torch_seed
from tokenizer import ReTokenizer
from data_loader import ReDataModule
from re_model_trainer import ReModelTrainer


def convert(config_path, config_file):

    fix_torch_seed()

    config = get_configuration(
        config_path=config_path,
        config_file=config_file)

    initialize_directories(config)

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

    model_list = os.listdir(config.path.model)
    model_fn = ""
    for model_name in model_list:
        if model_type in model_name:
            model_fn = model_name
            break

    path = os.path.join(config.path.model, model_fn)
    check_point = torch.load(path)
    re_model_trainer.load_state_dict(check_point["state_dict"])
    torch.save(
        re_model_trainer.model.state_dict(),
        os.path.join(config.path.hub, f"{model_type}.mdl"))


if __name__ == "__main__":
    convert(
        config_path="./",
        config_file="re.cfg"
    )