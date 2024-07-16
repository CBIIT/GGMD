FROM continuumio/miniconda3

# Add the necessary files
ADD spec-file.txt /tmp/spec-file.txt
ADD run_gmd.sh /run_gmd.sh
ADD app.py /app.py
ADD submit.sh /submit.sh

# Set executable permissions on scripts
RUN chmod a+x /run_gmd.sh

# Create conda environment
RUN conda create --name gmd --file /tmp/spec-file.txt

# Set environment variable
RUN echo "source activate gmd" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Install system dependencies for sendmail
RUN apt-get update && apt-get install -y sendmail

# Clone the repository
RUN git clone https://github.com/CBIIT/GGMD

# Install Flask (or any other dependencies you might need)
RUN pip install flask PyYAML
RUN conda install yaml -y

# Expose the port the app runs on
EXPOSE 8787

# Run the Flask app
CMD ["python", "/app.py"]

