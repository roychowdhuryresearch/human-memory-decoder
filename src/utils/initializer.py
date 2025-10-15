from src.dataloader.movie import *
from src.dataloader.free_recall import *
from src.models.multichannel_encoder_vit_mean2 import MultiEncoder as MultiEncoderViTMean2
from src.models.multichannel_encoder_vit_mean_logistic import PureLogisticRegression as MultiEncoderViTMeanLogistic
from src.models.multichannel_encoder_vit_mean_lfp import MultiEncoder as MultiEncoderViTMeanLFP
from src.utils.evaluator import Evaluator
from src.models.ensemble import *
import src.param.param_transformer as param_transformer
from src.param.param_data import *
from transformers import ViTConfig

def initialize_configs(architecture=''):
    if architecture == 'transformer':
        args = param_transformer.param_dict
    else:
        raise NotImplementedError(f'{architecture} is not implemented')
    return args

def initialize_inference_dataloaders(config):
    dataset = FreeRecallDataset(config)
    
    LFP_CHANNEL[config['patient']] = dataset.lfp_channel_by_region
    SPIKE_CHANNEL[config['patient']] = dataset.spike_channel_by_region # dataset.data.shape[2]
    test_loader = create_inference_combined_loaders(dataset, config, batch_size=config['batch_size'])

    dataloaders = {"train": None,
                   "valid": None,
                   "inference": test_loader}
    return dataloaders


def initialize_dataloaders(config):
    transform = None

    dataset = MovieDataset(config)
    LFP_CHANNEL[config['patient']] = dataset.lfp_channel_by_region
    SPIKE_CHANNEL[config['patient']] = dataset.spike_channel_by_region # dataset.data.shape[2]
    train_loader, val_loader, test_loader = create_weighted_loaders(
        dataset,
        config,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        p_val=0,
        transform=transform)

    dataloaders = {"train": train_loader,
                   "valid": val_loader,
                   "inference": test_loader}
    return dataloaders


def initialize_evaluator(config, fold):
    evaluator = Evaluator(config, fold)
    return evaluator


def initialize_model(config): 
    lfp_model = None
    spike_model = None

    if config['use_lfp']:
        if config['model_architecture'] == 'transformer':
            image_height = LFP_CHANNEL[config['patient']]
            image_height = list(image_height.values())
            image_width = LFP_FRAME[config['patient']]
            cfg = {
                "hidden_size": config['hidden_size'],
                "num_csa_layers": config['num_csa_layers'],
                "num_cca_layers": config['num_cca_layers'],
                "num_attention_heads": config['num_attention_heads'],
                # "intermediate_size": config['intermediate_size'],
                "image_height": image_height,
                "image_width": image_width,
                "patch_size": (1, 25),  # (height ratio, width)
                'input_channels': len(image_height),
                "num_labels": config['num_labels'],
                "num_channels": 1,
                "return_dict": True,
            }
            configuration = ViTConfig(**cfg)
            lfp_model = MultiEncoderViTMeanLFP(configuration)
        else:
            raise ValueError(
                f"Model Architecture {config['model_architecture']} not supported"
            )


    if config['use_spike']:
        # config.num_neuron = SPIKE_CHANNEL[config.patient]
        # config.num_frame = SPIKE_FRAME[config.patient]
        # config.return_hidden = True
        # spike_model = Wav2VecForSequenceClassification(config)
        if config['model_architecture'] == 'transformer':
            region_dict = SPIKE_CHANNEL[config['patient']]
            image_height = list(region_dict.values())
            image_width = SPIKE_FRAME[config['patient']]

            if config['use_overlap'] or config['use_long_input']:
                image_width = image_width * 2
            cfg = {
                "hidden_size": config['hidden_size'],
                "num_csa_layers": config['num_csa_layers'],
                "num_cca_layers": config['num_cca_layers'],
                "num_attention_heads": config['num_attention_heads'],
                # "intermediate_size": config['intermediate_size'],
                'region_dict': region_dict,
                'input_channels': sum(image_height) // 8,
                "image_height": 8,
                "image_width": image_width,
                "patch_size": config['patch_size'], #(1, 5),  # (height ratio, width)
                "num_labels": config['num_labels'],
                "num_channels": config['num_channels'],
                "return_dict": True,
            }
            configuration = ViTConfig(**cfg)
            # spike_model = ViTForImageClassification(configuration)
            if config['model_aggregate_type'] == 'mean2':
                spike_model = MultiEncoderViTMean2(configuration)
            elif config['model_aggregate_type'] == 'logistic':
                spike_model = MultiEncoderViTMeanLogistic(configuration)
            elif config['model_aggregate_type'] == 'lfp':
                spike_model = MultiEncoderViTMeanLFP(configuration)
            else:
                raise NotImplementedError(f'model aggregate type is not implemented')
        else:
            raise ValueError(
                f"Model Architecture {config['model_architecture']} not supported"
            )

    model = Ensemble(lfp_model, spike_model, config)
    return model
