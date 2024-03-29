# Sentiment Analysis Service Setup

## Project Description
Development of software for assessing the emotional state of Russian Railways employees using Python and sentiment analysis.

## Datasets

To train the sentiment analysis models, use the following datasets:

- **RuSentiment (180k records)**: Access the dataset [here](https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset).

- **Railway Sentiment (10k records)**: Access the dataset [here](https://github.com/Kbnch7/railsent/blob/main/output.csv).

## Training the Models

Before starting the model training process, set up a virtual environment and install the required dependencies:

1. Create a virtual environment in your project directory:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

3. Install the required libraries and frameworks from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

Training time and resource consumption vary depending on the model and training method (CPU or GPU). For accelerated training, install and configure the following:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

After successful training, place the trained models and tokenizers (if applicable) in the `models_data_180k` directory.

## Using Pre-trained Models

If you prefer to use already trained models, you can download them from [this link](https://drive.google.com/file/d/1DOsI-1_AcFU_fbAAlOY69nnXfnzL8M9Y/view?usp=sharing). Replace the contents of the `models_data_180k` directory with the files from the downloaded archive.

## Setting Up the Services

All services are persistent and require setup as follows:

1. Verify Docker and Docker Compose versions to avoid compatibility issues:
   - Check with `docker -v` and `docker-compose -v`.
   - Update if necessary. Refer to [Docker version 25.0.3](https://docs.docker.com/engine/release-notes/) and [Docker Compose version v2.24.6-desktop.1](https://docs.docker.com/compose/release-notes/).

2. Navigate to the `Model_Services` directory.
3. Start the services using Docker:
   ```
   docker-compose up --build
   ```

## Running the Services

Once the services are up, the interface will be available at [http://localhost:7860/](http://localhost:7860/). Ensure port 7860 is free before launching.

The services for interacting with the model will be ready when the server enters debug mode, indicated by the message `Debugger is active! | Debugger PIN: 973-382-319`.
