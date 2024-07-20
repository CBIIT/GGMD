# AMPL Containers Need for this Demo:
For this demo, pull an AMPL container:

Docker: 
`docker pull atomsci/atomsci-ampl`

Singularity:
`singularity build ampl.sif docker://atomsci/atomsci-ampl`

While 

# About this example

In this example, we are demonstrating the application of a predictive model as the scoring function for the GGMD loop. In this particular example, we have implemented the Atom Modeling Pipeline (AMPL) as the software for the predictive model.

AMPL is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery. To learn more about AMPL, please refer to the [GitHub page here](https://github.com/ATOMScience-org/AMPL) to learn more. 

Additionally, it may be useful to review the example located at `examples/LogP_JTVAE` first. That example describes the JTVAE method and how GMD works with JTVAE. While this example applies the JTVAE method, the focus is on the application of an AMPL predictive model as the scorer function.

To reproduce this test or adapt to your problem, please follow the below steps:
1. If you intend to use a predictive model, you will first need to train that model on a dataset.
2. In the `config.yaml` file, please be sure to edit all of the necessary variables.
3. If you plan to use singularity, be sure that your working directory is set to be a location where you have root access. Shared directories often won't work.
4. In the file `examples/AMPL_model_JTVAE/predict_from_model_file.py`, please change line 6 and replace the text `model_file_name` with the name of your model file. An example would look like the follow:

    `model_file = "/data/model_file.tar.gz"`

### Important Notes

- you will see in different locations that some file paths in our code start with /data. This is specific to the file structure in the container being used. When we execute the container to run the inference, we mount our working director to the countainer. This allows us to pass files into the container and extract files that are created in the container such as the population csv file. In the singularity/docker execute commands in the scorer, we mount the working directory (as defined by the user in the config file) to the folder `/data` inside the container. *Please do not change that file path. Doing so may stop the code from working*.
- While this example uses the JTVAE method to manipulate the molecules, our implementation of AMPL for the scorer (which is the focus of this example) is not specific to the JTVAE method. You can apply this scorer, in the same way, to the AutoGrow method implemented by simply changing out the generative model parameters found in the config file. Please refer to the AutoGrow example (`examples/AMPL_model_JTVAE`) for an example config file.


The implemented scorer class for AMPL predictive models can be found at `source/scorers/AMPL_pred_model/ampl_pred_model.py