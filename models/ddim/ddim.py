import numpy as np
import torch as th
from torch.nn import functional as F
from models.munet import MemoryUnet


def get_beta(betas_schedule="linear", num_diffusion_timesteps=1000):
    assert 100 <= num_diffusion_timesteps <= 1000
    if betas_schedule == "linear":
        beta_start = 0.0001
        beta_end = 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {betas_schedule}")


class DDIM:
    def __init__(
            self,
            *,
            betas_schedule="linera",
            time_steps=1000,
            ddim_step=100,
    ):
        betas = get_beta(betas_schedule, time_steps)
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas

        self.time_steps = time_steps
        self.ddim_step = ddim_step

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.time_steps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: The [N x C x ...] Tensor of noiseless inputs.
        :param t: The number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
        后验均值和方差
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """
        神经网络得到逆过程的均值和方差
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(x_t, t)

        # 固定方差
        model_variance, model_log_variance = (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:])),
        )
        model_variance = _extract_into_tensor(model_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )   # 由x0得到，xt得到xt-1分布的均值和方差

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(self, model, x_t, t, clip_denoised=True):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x_t: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # 算出x_{t-1}的均值和方差
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
        )
        noise = th.randn_like(x_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        sample_list = []
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                device=device,
                progress=progress,
        ):
            final = sample
            sample_list.append(final["sample"])
        return final["sample"], sample_list

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.time_steps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x_t,
            t,
            t_next,
            clip_denoised=True,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
        )
        eps = self._predict_eps_from_xstart(x_t, t, out["pred_xstart"])  # get eps through predict_xstart
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod, t_next, x_t.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_next) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_next)
        )  # variance
        # Equation 12.
        noise = th.randn_like(x_t)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next - sigma ** 2) * eps
        )   # predicted mean
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        sample_list = []
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
            sample_list.append(final["sample"])
        return final["sample"], sample_list

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(th.linspace(0, self.time_steps - 1, self.ddim_step + 1).long())[::-1]
        t_pairs = zip(indices[:-1], indices[1:])
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            t_pairs = tqdm(t_pairs, total=self.ddim_step)

        for i, j in t_pairs:
            t = th.tensor([i] * shape[0], device=device)
            t_next = th.tensor([j] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    t_next,
                    clip_denoised=clip_denoised,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: loss.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, t)
        target = noise
        assert model_output.shape == target.shape == x_start.shape
        loss = F.mse_loss(noise, model_output)
        return loss

    def get_rand_t(self, batch_size, device):
        t1 = th.randint(0, self.time_steps // 2, (batch_size // 2,)).long()
        t2 = self.time_steps - 1 - t1
        t3 = th.cat([t1, t2], dim=-1)
        return t3.to(device)


def _extract_into_tensor(arr, t, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param t: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=t.device)[t].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def create_ddim_and_unet(device):
    diff = DDIM(
        betas_schedule="linear",
        time_steps=1000,
        ddim_step=100
    )
    unet = MemoryUnet(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[],
        num_res_blocks=2,
        MEM_DIM=200,
        addressing="sparse",
    ).to(device)
    return diff, unet


