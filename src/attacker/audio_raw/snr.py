import torch


def calculate_snr(clean_audio: torch.Tensor, audio_noised: torch.Tensor) -> torch.Tensor:
    """Compute SNR in dB using the prepend-attack assumption.

    Args:
        clean_audio: Reference audio with a leading silent segment (zeros) that
            aligns with the attack duration. Shape [..., time].
        audio_noised: Audio that includes the prepended attack segment. Same
            shape as ``clean_audio``.

    Returns:
        A tensor of SNR values in dB for each item in the batch (or a scalar for
        single vectors).
    """

    squeeze_back = False
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)
        audio_noised = audio_noised.unsqueeze(0)
        squeeze_back = True

    signal_power = (clean_audio ** 2).sum(dim=1)
    noise_power = ((clean_audio - audio_noised) ** 2).sum(dim=1)

    snr = torch.where(
        noise_power == 0,
        torch.full_like(signal_power, float("inf")),
        10 * torch.log10(signal_power / noise_power),
    )

    if squeeze_back:
        snr = snr.squeeze(0)
    return snr
