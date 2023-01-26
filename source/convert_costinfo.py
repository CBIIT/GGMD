import numpy as np

import logging
Log = logging.getLogger(__name__)


def convert_costinfo_dataframe_to_dict(df):
    """Converts a costinfo DataFrame to a dict.

    used for converting (legacy) CSV costinfo definitions
    to YAML/dictionary format

    """
    colmap = {
        'Name': 'name',
        'Description': 'description',
        'Framework': 'framework',
        'Location': 'location',
        'ID': 'id',
        'Allow_neg': 'allow_neg',
        'CostFunction': 'cost_function',
        'Target_min': 'target_min',
        'Target_max': 'target_max',
        'Weight': 'weight',
        'Scale': 'scale',
    }

    top_level = [
        'model_name',
        'description',
        'framework',
        'location',
        'id',
    ]

    cost_terms = [
        'name',
        'description',
        'output_type',
        'cost_function',
        'target_min',
        'target_max',
        'allow_neg',
        'weight',
        'scale',
        'is_log',
        'features',
        'score_agg_func'
    ]

    df = df.fillna(np.nan).replace([np.nan], [None])
    df.rename(columns=colmap, inplace=True)

    # get docking/fusion rows and dataframes that are defined with receptor_id
    docking_rows = df['name'].str.startswith('docking_score_')
    fusion_rows = df['name'].str.startswith('fusion_score_')
    docking_df = df[docking_rows]
    fusion_df = df[fusion_rows]

    # remove docking/fusion receptor_id rows from dataframe
    df = df.loc[~(docking_rows | fusion_rows)]

    dd = df.to_dict('records')

    def update_entry(entry):
        entry['model_name'] = entry['name']
        new_entry = { k:entry[k] for k in top_level }
        for k in top_level:
            ncost_terms = {k: entry.get(k) for k in cost_terms}
            if ncost_terms['output_type'] is None:
                ncost_terms['output_type'] = 'prediction'

            new_entry['cost_terms'] = [ncost_terms]

        return new_entry

    # all models except for receptor_id docking/fusion
    models = [update_entry(d) for d in dd]

    # add docking/fusion models
    docking_dd = docking_df.to_dict('records')
    fusion_dd = fusion_df.to_dict('records')

    def update_cost_terms(entry, mtype, rstr):
        receptor_id = entry['name'].replace(rstr, '')
        ncost_terms = {k: entry.get(k) for k in cost_terms}
        ncost_terms['name'] = receptor_id
        ncost_terms['receptor_id'] = "${name}"
        ncost_terms['description'] = (
            f"{mtype} scoring for receptor id: " + "${receptor_id}"
        )
        ncost_terms['output_type'] = 'prediction'
        ncost_terms['score_agg_func'] = 'min' if mtype == 'docking' else 'max'
        ncost_terms['model_result_id'] = '${model_name}_${receptor_id}'
        return ncost_terms

    docking_model = {
        'model_name': 'docking_score',
        'description': 'docking scoring',
        'framework': 'fusion',
        'location': 'fusion',
        'id': 'fusion',
        'cost_terms': [
            update_cost_terms(d, mtype='docking', rstr='docking_score_')
            for d in docking_dd
        ]
    }

    if docking_model['cost_terms']:
        models.extend([docking_model])

    fusion_model = {
        'model_name': 'fusion_score',
        'description': 'fusion scoring for receptors',
        'framework': 'fusion',
        'location': 'fusion',
        'id': 'fusion',
        'cost_terms': [
            update_cost_terms(d, mtype='fustion', rstr='fusion_score_')
            for d in fusion_dd
        ]
    }

    if fusion_model['cost_terms']:
        models.extend([fusion_model])

    odd = {
        'models': models
    }

    return odd


def convert_costinfo_csv_to_yaml_main():
    """main for converting CSV to YAML costinfo files"""

    import argparse
    from collections import OrderedDict

    from .costinfo import CostInfo

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--infile", dest="infile", required=True,
        help="Input costinfo CSV file to convert"
    )
    parser.add_argument(
        "-o", "--outfile", dest="outfile", required=True,
        help="The converted output costinfo YAML file"
    )

    args = parser.parse_args()

    print(f"[*] reading costinfo CSV file: {args.infile}")

    costinfo = CostInfo(args.infile)

    valid, err_str = costinfo.validate()

    print(f"[*] converted {costinfo.model_count} model entries.")

    costinfo.save(args.outfile)

    print(f"[*] generated YAML cost info file: {args.outfile}")

    if not valid:
        print(f"\nWARNING: converted YAML has errors")
