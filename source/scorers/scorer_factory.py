

def create_scorer(params):
    if params.scorer_type == 'LogPTestCase':
        from scorers.LogPOctanol.logp_octanol_scorer import LogPOctanolWaterPartitionCoef
        return LogPOctanolWaterPartitionCoef(params)
    else:
        raise ValueError("Unknown scorer_type: %s" % params.scorer_type)