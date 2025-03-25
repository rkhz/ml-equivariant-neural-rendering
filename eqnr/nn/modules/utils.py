def get_num_groups(
    num_channels: int
    ) -> int:
    """Returns number of groups to use in a GroupNorm layer with a given number
    of channels. Note that these choices are hyperparameters.

    Args:
        num_channels (int): Number of channels.
    """

    thresholds = [8, 32, 64, 128, 256]
    num_groups = [1, 2, 4, 8, 16, 32]

    return num_groups[sum(num_channels >= t for t in thresholds)]