import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .cell_model import  Cell_Unet


def diffusion_defaults(diffusion_steps=1000):
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=diffusion_steps,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults(diffusion_steps=1000):
    """
    Defaults for image training.
    """
    res = dict(
        input_dim = 128,
        hidden_dim = [512,512,256,128],
        dropout = 0.0,
    )
    res.update(diffusion_defaults(diffusion_steps=diffusion_steps))
    return res


def create_model_and_diffusion(
    input_dim,
    hidden_dim,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    dropout,
    num_classes,
    cache_interval=None,
    non_uniform=None,
    branch=None,
    **kwargs,
):
    model = create_model(
        input_dim,
        hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        cache_interval = cache_interval,
        non_uniform = non_uniform,
        branch = branch,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    input_dim,
    hidden_dim,
    num_classes,
    dropout,
    cache_interval,
    non_uniform,
    branch,
):

    return Cell_Unet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        cache_interval = cache_interval,
        non_uniform = non_uniform,
        branch = branch,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
