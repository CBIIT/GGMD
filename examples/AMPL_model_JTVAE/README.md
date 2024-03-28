# About this example

In this example, we are demonstrating the application of a predictive model as the scoring function for the GGMD loop. In this particular example, we have implemented the Atom Modeling Pipeline (AMPL) as the software for the predictive model.

AMPL is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery. To learn more about AMPL, please refer to the [GitHub page here](https://github.com/ATOMScience-org/AMPL) to learn more.

If you intend to use a predictive model, you will first need to train that model on a dataset. In the `config.yaml` file, please be sure to edit the appropriate scorer specific variables at the bottom. Additionally, if you plan to use singularity, be sure that your working directory is set to be a location where you have root access. Shared directories often won't work.

The implemented scorer class for AMPL predictive models can be found at `source/scorers/AMPL_pred_model/ampl_pred_model.py