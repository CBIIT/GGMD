


def create_generative_model(params):
    """
    Factory function for creating optmizer objects of the correct subclass for params.optmizer_type.
    Args:
        params: parameters to pass
    Returns:
        optmizer (object):  wrapper
    Raises: 
        ValueError: Only params.VAE_type = "JTNN" is supported
    """

    if params.model_type.lower() == 'jtnn-fnl':
        from generative_networks.JTNN import JTNN_FNL

        return JTNN_FNL(params)
    elif params.model_type.lower() == 'autogrow':
        from generative_networks.AutoGrow import AutoGrow

        return AutoGrow(params)
    else:
        raise ValueError("Unknown model_type %s" % params.model_type)