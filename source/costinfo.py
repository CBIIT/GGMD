import copy
from string import Template
from collections import namedtuple
import logging
from dataclasses import dataclass

from mergedeep import merge
import pandas as pd
import yaml

from convert_costinfo import convert_costinfo_dataframe_to_dict


Log = logging.getLogger(__name__)


# Wraps a given model entry in costinfo CSV/YAML files
@dataclass
class ModelCostInfo:
    model_name: str  # the model name
    description: str  # the model description
    framework: str  # the model framework
    id: str  # the model ID (uuid)
    location: str  # the model state location (path)
    cost_terms: list  # the list of CostTerms cost terms items


class CostTerms(object):
    """Wraps each cost_terms item for models in costinfo CSV/YAML files.

    Valid cost_terms fields:
    ------------------------
    model_name (str): the model name associated with the cost terms
    idx (int): the index of the cost terms item
    name (str): the name of the cost_terms item
    description (str): the description
    output_type (str): the selected output type of the model predictor to
       use for these cost terms
    cost_function (str): the function to use when computing cost. Must be one
       of 'exp', 'linear', 'binary'. (see Scorer)
    target_min (float or None): the target minimum used for rectification in
       the cost function
    target_max (float or None): the target maximum used for rectification in
       the cost function
    scale (float): scale used in cost function with target_min/max (default=1.0)
    allow_neg (bool): whether or not to allow negative cost in cost function
    weight (float): cost multiplier (default=1.0)
    is_log (bool): unused ???
    features (str): unused ???
    receptor_id (str): the receptor identifier used for prediction
    score_agg_func (str): the aggregation function to use when there are
       multiple prediction values for a given compound. Currently
       only 'min', 'max' are supported
    model_result_id (str): The unique model result identifier to be used with
       this cost terms item. Must be set by the model and used by the
       model aggregator as a column heading for model results.
       (default: '${model_name}')

    Supports Template variable substitution of cost_term field values when
    retrieving the follwoing class properties:
    - name
    - description
    - receptor_id
    - model_result_id
    """

    valid_cost_functions = ['exp', 'linear', 'binary']

    def __init__(self, model_name, idx, raise_exc=False, **kwargs):
        self.model_name = model_name
        self.idx = idx
        self._name = kwargs.pop('name', 'unknown')
        self._description = kwargs.pop(
            'description', '${name} cost terms for model ${model_name}'
        )
        self.output_type = kwargs.pop('output_type', 'prediction')
        self.cost_function = kwargs.pop('cost_function', None)
        self.target_min = kwargs.pop('target_min', None)
        self.target_max = kwargs.pop('target_max', None)
        self.allow_neg = kwargs.pop('allow_neg', False)
        self.weight = kwargs.pop('weight', 1.0)
        self.scale = kwargs.pop('scale', 1.0)
        self.is_log = kwargs.pop('is_log', 0)
        self.features = kwargs.pop('features', None)
        self._receptor_id = kwargs.pop('receptor_id', None)
        self._score_agg_func = kwargs.pop('score_agg_func', None)
        self._model_result_id = kwargs.pop(
            'model_result_id', '${model_name}:${idx}'
        )

        if self.cost_function not in self.valid_cost_functions:
            err_str = (
                f"Model {model_name} has invalid cost function: "
                f"{self.cost_function} [valid={self.valid_cost_functions}]"
            )
            if not raise_exc:
                Log.warn(err_str)
            else:
                Log.error(err_str)
                raise Exception(err_str)

        if len(kwargs):
            err_str = (
                f"Model {model_name} has unknown cost terms: "
                f"name={self.name}: unknown terms (idx={idx}): {kwargs}"
            )
            if not raise_exc:
                Log.warn(err_str)
            else:
                Log.error(err_str)
                raise Exception(err_str)

    def as_dict(self, substitute=True):
        vals = vars(self).copy()
        nvals = {k[1:]: v for k, v in vals.items() if k.startswith('_')}
        vals.update(nvals)
        vals = {k: v for k, v in vals.items() if not k.startswith('_')}
        if substitute:
            vals = {k: self._substitute(v) for k, v in vals.items()}
        return vals

    def _substitute(self, strval):
        if not strval or not isinstance(strval, str):
            return strval

        vals = self.as_dict(substitute=False)
        while True:
            nstrval = Template(strval).safe_substitute(
                **vals
            )
            if strval != nstrval:
                strval = nstrval
            else:
                return nstrval

    def __str__(self):
        vals = self.as_dict()
        return str(vals)
        # strval = yaml.dump(vals, sort_keys=False)
        # return strval

    def __repr__(self):
        vals = self.as_dict(substitute=True)
        return str(vals)
        # strval = yaml.dump(vals, sort_keys=False)
        # return strval

    @property
    def name(self):
        return self._substitute(self._name)

    @property
    def description(self):
        return self._substitute(self._description)

    @property
    def receptor_id(self):
        return self._substitute(self._receptor_id)

    @property
    def model_result_id(self):
        return self._substitute(self._model_result_id)

    @property
    def score_agg_func(self):
        if self._score_agg_func == 'max':
            return max
        elif self._score_agg_func == 'min':
            return min
        else:
            return None


class CostInfo(object):
    """Class wrapper for managing and interfacing to costinfo CSV/YAML files.

    YAML Top-level fields
    ---------------------
    model_name (str): the model name
        needs to match model_predictor worker name
    description (str): the model description
    framework (str): the model framework. Valid values are:
        ATOM_file, ATOM, ATOM_mlmt: (ATOM AMPL based frameworks)
        features: use features
        rdkit.Chem: rdkit based
        pytorch: (OBSOLETE) neurocrine hybrid loss
    id (str): the unique identifier for the model (uuid)
    location (str): location of model category or state for frameworks:
        - atom models this is the model state location (filename) or otherwise
          required by the specific AMPL model
        - features this currently must be 'MOE'
        - rdkit based this is the category of rdkit.Chem models and is one of
          'Descriptors', 'SyntheticAccessibility', 'Lipinski', 'PAINS', 'Crippen'
    cost_terms (list(CostTerms)): defines scoring cost terms for the model
        see CostTerms class
    """
    def __init__(self, costinfo_filename):
        """costinfo_filename can be either CSV or YAML file

        if CSV file, then converts (legacy) to YAML dictionary object
        """
        self.load(costinfo_filename)

    def load(self, fname=None):
        """Load the costinf CSV/YAML file"""
        fname = fname or self.filename
        self.filename = fname
        if self.filename.endswith(".csv"):
            df = pd.read_csv(self.filename)
            d = convert_costinfo_dataframe_to_dict(df)
        else:
            d = yaml.load(open(self.filename, 'r'), yaml.SafeLoader)

        self.costinfo_config = d.copy()

        # get list of models from costinfo dictionary
        models = d['models']

        # index costinfo on model_name
        self.costinfo = {m['model_name']: m for m in models}
        return self.costinfo

    def save(self, filename, substitute=True):
        if not substitute:
            yaml.dump(
                self.costinfo_config, open(filename, 'w'), yaml.SafeDumper,
                sort_keys=False
            )
            return

        d = {k: self.get_model_costinfo_as_dict(k) for k in self.costinfo}
        od = {'models': d}

        yaml.dump(od, open(filename, 'w'), yaml.SafeDumper, sort_keys=False)

    @property
    def model_names(self):
        return list(self.costinfo.keys())

    def validate(self, model_names=None):
        valid = True
        errors = []

        model_names = model_names or self.model_names
        for model_name in model_names:
            try:
                self.get_model_costinfo(model_name, raise_exc=True)
            except Exception as e:
                valid = False
                errors.append(str(e))

        return (valid, "\n".join(errors))

    def get_model_costinfo(self, model_name, raise_exc=False):
        if model_name not in self.costinfo.keys():
            raise KeyError(
                f"Model name: {model_name} not specified in costinfo "
                f"file: {self.filename}"
            )

        m = self.costinfo[model_name]
        cost_terms = m.get('cost_terms')
        if not cost_terms:
            raise ValueError(
                f"Model name: {model_name} does not have any cost_terms"
            )

        model_costinfo = ModelCostInfo(
            model_name=m['model_name'],
            description=m.get('description', "undefined"),
            framework=m.get('framework', 'unknown'),
            location=m.get('location', 'unknown'),
            id=m.get('id', 'unknown'),
            cost_terms=[
                CostTerms(m['model_name'], idx, raise_exc=raise_exc, **ct)
                for idx, ct in enumerate(m.get('cost_terms', []))
            ]
        )

        return model_costinfo

    def get_model_costinfo_as_dict(self, model_name):
        ci = self.get_model_costinfo(model_name)
        cost_terms = [ct.as_dict() for ct in ci.cost_terms]
        ci.cost_terms = cost_terms
        return vars(ci)

    def model_iter(self, limit=None):
        limit = limit or {}
        miter = self.costinfo.values()

        # limit iteration based on model_name
        limit_model_names = limit.get('model_names')
        if limit_model_names:
            miter = filter(
                lambda d: d['model_name'] in limit_model_names, miter
            )

        return miter

    @property
    def model_count(self):
        return len(self.costinfo)


class CostScoringConfig(object):
    """Manages access to cost_scoring_config property configuration dictionary.

    1) configures which model predictor workers are being used
       - includes ATOM, rdkit, docking and fusion derived model predictors
       - specifies resources (job_spec) for running each worker
       - specifies how many model results (one per message to model aggregator)
         are generated by each model
    2) the configured model predictors must:
       - be a subset of those specified in the costinfo file/config
       - have defined receptors that are a subset of those defined in
         the costinfo

    The configuration has these top-level fields:
    ---------------------------------------------
        model_predictor_defaults (dict): default config values for
           model_predictors
        receptors (list(str)): list of receptors (optional)
        model_predictors (list(dict)): list of model_predictors configurations

    model_predictor_defaults
    ------------------------
    1) specifies the default config values for model_predictors if not
       explicitely specified in the model_predictor config
    2) all fields are optional and extra fields can be added/used
    3) the following fields are currently used:
       worker_template (str): defines the worker template to use to
          instantiate the model_predictor (default='model_predictor')
          if None, then no template generation is done and the model predictor
          is not dynamically launched via flux adapter
       use_flux_adapter (bool): if true, launch worker via flux adapter
          (default=true)
       worker_batch_count (int): the number of batches to process before
          a model_predictor exits. Currently this must be 1. If set to 0,
          the worker will not exit (currently not supported though).
       result_count (int): the number of results to expect from this
          model predictor (used when aggregating model results)
       job_spec (dict): the job spec for running the model_predictor
          only used if worker_termplate is specified and worker is launched
          dynamically via a flux adapter
          fields: num_tasks, num_nodes, cores_per_task, gpus_per_task, urgency
            see flux job_spec documentation
            urgency is a float between 0.0 and 1.0 and is converted to
            a flux_urgency when queing the worker.
    """
    def __init__(self, cost_scoring_config):
        cost_scoring_config = cost_scoring_config or {}
        self.cost_scoring_config = copy.deepcopy(cost_scoring_config)
        self.model_predictor_defaults = cost_scoring_config.get(
            'model_predictor_defaults'
        )
        self.receptors = cost_scoring_config.get('receptors', [])
        self.model_predictors = {
            x['model_name']: self.update_model_defaults(x)
            for x in cost_scoring_config.get('model_predictors', [])
        }
        self.model_result_count = sum([
            x.get('result_count', 1) for x in self.model_predictors.values()
        ])

    @property
    def model_names(self):
        return self.model_predictors.keys()

    def update_model_defaults(self, model_config):
        """update model config with defaults from scoring config

        does varible substitutione on the following:
        - result_count

        defined variables:
        - ${receptor_count} (int): the number of receptors
        """
        # apply defaults to model_predictor config
        model_config = merge({}, self.model_predictor_defaults, model_config)

        # do variable substitution on 'result_count'
        result_count = model_config.get('result_count')
        if isinstance(result_count, str):
            result_count = int(Template(result_count).safe_substitute(
                receptor_count=len(self.receptors)
            ))
            model_config['result_count'] = result_count if result_count else 1

        return model_config

    def save(self, filename):
        d = {
            'cost_scoring_config': self.cost_scoring_config,
            'model_names': list(self.model_names),
            'receptors': list(self.receptors),
            'model_result_count': int(self.model_result_count),
            'model_predictors': [
                v for v in self.model_predictors.values()
            ]
        }
        yaml.dump(d, open(filename, 'w'), yaml.SafeDumper, sort_keys=False)

    def validate(self, costinfo):
        valid = True
        errors = []

        # validate costinfo has all the cost scoring configured models
        model_names = set(self.model_names)
        valid, err_str = costinfo.validate(model_names=model_names)
        if not valid:
            errors.append(err_str)

        # check for missing model names
        missing_models = model_names - set(costinfo.model_names)
        if missing_models:
            valid = False
            mstr = ", ".join(missing_models)
            errors.append(
                f"cost info file {costinfo.filename} does not include scoring "
                f"information for the following models model: {mstr} "
            )

        # check for receptor_ids that are specified for scoring but
        # are not defined in the costinfo file
        ct_receptors = []
        for model_name in model_names:
            model_costinfo = costinfo.get_model_costinfo(model_name)
            for cost_terms in model_costinfo.cost_terms:
                receptor_id = cost_terms.receptor_id
                if receptor_id:
                    ct_receptors.append(receptor_id)

        # ensure
        missing_receptors = set(ct_receptors) - set(self.receptors)
        if missing_receptors:
            valid = False
            mstr = ", ".join(missing_receptors)
            errors.append(
                f"costinfo file {costinfo.filename} does not include scoring "
                f"definition for receptors: {mstr}"
            )

        return (valid, "\n".join(errors))
