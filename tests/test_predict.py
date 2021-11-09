import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from re_config import get_configuration
from prediction_model import PredictionModel

def main(config_path, config_file):
    config = get_configuration(
        config_path=config_path,
        config_file=config_file)

    model = PredictionModel(config=config)
    result = model.predict(
        sentence="가브리엘 창손 창()은 남수단의 정치인이다.",
        subj_start_idx=0,
        subj_end_idx=9,
        subj_label="Person",
        obj_start_idx=18,
        obj_end_idx=21,
        obj_label="Civilization",
        return_method="prob"
    )

    print(result)

if __name__ == "__main__":
    main(
        config_path="./",
        config_file="re_test.cfg"
    )