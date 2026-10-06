from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    fc_zscore_within_subject: bool = False
    fc_input_mode: str = "continuous"
    fc_binarization_method: str = "subject_mean_01"
    fc_continuous_standardize: bool = True
    std_epsilon: float = 1e-6


@dataclass
class AutoencoderConfig:
    latent_dim: int = 120
    hidden_1: int = 2048
    hidden_2: int = 512
    dropout: float = 0.03
    train_batch_size: int = 32
    latent_batch_size: int = 128
    num_workers: int = 0
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    pretrain_lr: float = 5e-4
    finetune_lr: float = 1e-4
    bin_weight_max: float = 0.01
    bal_weight_max: float = 0.005
    sep_weight: float = 0.002
    use_correlation_loss: bool = True
    corr_loss_weight: float = 0.10
    pretrain_noise_std: float = 0.05
    finetune_noise_std: float = 0.02
    pretrain_tanh_start: float = 0.5
    pretrain_tanh_end: float = 0.8
    finetune_tanh_start: float = 0.8
    finetune_tanh_end: float = 1.5
    pretrain_patience: int = 12
    finetune_patience: int = 12
    finetune_reg_warmup_fraction: float = 0.35
