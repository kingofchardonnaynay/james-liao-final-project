# Pokemon TCG Identifier Project

## Table of Contents
- [Project Description](#project-description)
- [App Link](#app-link)
- [Features](#features)
- [Files and Directories](#files-and-directories)
- [License](#license)

## Project Description
The **Pokemon TCG Identifier Project** is an application designed to identify Pokémon Trading Card Game (TCG) cards using image recognition. This tool allows users to upload images of Pokémon cards, and the app will return the card's set name, collector number, and Pokémon card name.

The project uses advanced image processing and machine learning techniques, including convolutional neural networks (CNNs) and Optical Character Recognition (OCR), to extract key information from each card.

## App Link
The application is hosted locally as the model is too large for github.
model_utils.py has a link to the model's location on Google Cloud but authentication has been a problem.

## Features
- Upload a Pokémon TCG card image and receive detailed information about the card.
- Uses CNNs to predict the set symbol of the card.
- Applies OCR to extract the card’s collector number and total card count.
- Two methods are attempted to identify the card: a combination of set symbol + collector number, or simply using the collector number
- Interactive web interface built with Streamlit.
- Supports identifying cards from multiple sets.


## Files and Directories
Here's an overview of the main files and directories used in the project:

- `streamlit_app.py`: The main file to run the Streamlit app.
- `packages.txt`: installs libgl1, a library that provides OpenGL functionality
- `src/`: Directory containing core application code, modules, etc.
- `documents/`: Contains initial proposal document
- `notebooks/`: Contains notebook training of model (retrieving data set, creating data set, training, etc.)
- `requirements.txt`: Contains a list of dependencies required to run the project.
- `README.md`: Documentation for the project.
- `data/`: Contains mandatory data files
- `Superseded/`: Directory containing other notebooks and py code that was ultimately discarded

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
