import os
import pickle
import time

import dotenv
import open_clip
import torch
import yaml
from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionLatentUpscalePipeline,
                       StableDiffusionPipeline)
from open_clip.factory import get_model_config, load_checkpoint
from open_clip.pretrained import (download_pretrained, get_pretrained_cfg,
                                  list_pretrained_tags_by_model)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

dotenv.load_dotenv()

MODULE_TYPE_CLASSES = {
    "vae": "diffusers.AutoencoderKL",
    "unet": "diffusers.UNet2DConditionModel",
    "feature_extractor": "transformers.CLIPFeatureExtractor",
    "tokenizer": "transformers.CLIPTokenizer",
    "text_encoder": "transformers.CLIPTextModel",
}

PIPELINE_CLASSES = {
    "sd_model": StableDiffusionPipeline,
    "upscaler": StableDiffusionLatentUpscalePipeline,
}


def get_half_modules(modules):
    for key, module in modules.items():
        try:
            halfed_module = module.half()
        except:
            print("failed to convert", key)
        else:
            modules[key] = module.half()
            print("converted", key)
    return modules

def save_modules(save_folder, modules, model_id):
    for key, module in modules.items():
        module_folder = os.path.join(save_folder, f"{model_id}")
        if not os.path.exists(module_folder):
            os.makedirs(module_folder)
        save_path = os.path.join(module_folder, f"{key}_half.pt")
        torch.save(module, save_path)
        print("saved to", save_path)
        
class ModelSaver:
    def __init__(
        self, 
        save_folder: str = "safetensors_models",
        sd_cfg_file: str = "sd_models_cfg.yaml",
        hf_token: str = os.getenv('HUGGING_FACE_TOKEN'),
    ):
        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            trained_betas=None,
            # predict_epsilon=True,
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
        )

        self._default_pipeline_options = {
            # "use_auth_token": hf_token, 
            "scheduler": scheduler,
            "safety_checker": None,
        }
        
        self._models = {}
        self.hf_token = hf_token
        self.sd_cfg_file = sd_cfg_file
        self.save_folder = save_folder
        with open(sd_cfg_file, "r") as cfg:
            models_cfg = yaml.load(cfg, Loader=Loader)
            self.models_cfg = models_cfg
        
    def save_all_models(self):
        for model_cfg in self.models_cfg:
            model_id = model_cfg.get("id", None)
            print(f"Loading model {model_id} from {self.sd_cfg_file}")
            assert model_id is not None, f"Error: model id is not defined in {self.sd_cfg_file}"                
            assert model_id not in self._models, f"Error: model {model_id} is already defined in {self.sd_cfg_file}"

            model_path = model_cfg.get("model_path", None)
            assert model_path is not None, f"Error: model_path is not defined in {self.sd_cfg_file}"
            saved_modules_path = os.path.join(self.save_folder, f"{model_id}")

            if not os.path.exists(saved_modules_path):
                pipeline_class = PIPELINE_CLASSES[model_cfg.get("pipeline_class", "sd_model")]
                print(f"Loading pipeline {pipeline_class} from {model_path}")
                dummy_pipe = pipeline_class.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        use_auth_token=self.hf_token,
                        **self._default_pipeline_options if pipeline_class != StableDiffusionLatentUpscalePipeline else {},
                    )
                dummy_pipe.save_pretrained(saved_modules_path, safe_serialization=True)
                del dummy_pipe
        
# def save_clip_model(clip_model, clip_pretrain, device):
#     model_cfg = get_model_config(clip_model)
#     model = CLIP(**model_cfg, cast_dtype=torch.float32).to(device)

#     image_mean = getattr(model.visual, 'image_mean', None)
#     image_std = getattr(model.visual, 'image_std', None)

#     preprocess = open_clip.transform.image_transform(
#         model.visual.image_size,
#         is_train=False,
#         mean=image_mean,
#         std=image_std
#     )

#     pretrained_cfg = {}
#     if clip_pretrain:
#         checkpoint_path = ''
#         pretrained_cfg = get_pretrained_cfg(clip_model, clip_pretrain)
#         if pretrained_cfg:
#             checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=None)
#         elif os.path.exists(clip_pretrain):
#             checkpoint_path = clip_pretrain

#         if checkpoint_path:
#             print(f'Loading pretrained {clip_model} weights ({clip_pretrain}).')
#             load_checkpoint(model, checkpoint_path)
#         else:
#             error_str = (
#                 f'Pretrained weights ({clip_pretrain}) not found for model {clip_model}.'
#                 f'Available pretrained tags ({list_pretrained_tags_by_model(clip_model)}.')
#             print(error_str)
#             raise RuntimeError(error_str)

#     return model, preprocess
        
        


if __name__ == "__main__":
    model_folder = os.getenv("MODEL_FOLDER", '/krea-models')
    # model_folder="./safetensor-models"
    print(f"saving models to {model_folder}")
    model_loader = ModelSaver(save_folder=model_folder)
    model_loader.save_all_models()
