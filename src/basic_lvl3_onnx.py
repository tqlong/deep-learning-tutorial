import os
import rootutils
import hydra
import torch
from omegaconf import DictConfig
import logging
logging.basicConfig(level=logging.INFO)

# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")

from src.basic_lvl3 import TransferLearningModule


@hydra.main(version_base=None,
            config_path=config_path,
            config_name="basic_lvl3_onnx")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    model: TransferLearningModule = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(cfg.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    filepath = os.path.join(cfg.paths.output_dir, cfg.model_onnx_name)
    logging.info(f"model output {filepath}")
    input_sample = torch.randn((1, 3, 32, 32))
    model.to_onnx(filepath, input_sample, export_params=True)

    # test onnx model
    import onnxruntime
    import numpy as np

    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 3, 32, 32).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    logging.info(f"onnx output: {ort_outs}")

    # save torchscript model
    script = model.to_torchscript()
    script_path = os.path.join(cfg.paths.output_dir, cfg.model_torchscript_name)
    torch.jit.save(script, script_path)
    # test torchscript model
    scripted_module = torch.jit.load(script_path)
    inp = torch.randn((1, 3, 32, 32), dtype=torch.float32)
    output = scripted_module(inp)
    logging.info(f"torch script output: {output}")


if __name__ == "__main__":
    main()
