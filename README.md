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
You can access the live application here: [Pokemon TCG Identifier](#)

(Note: Replace `#` with the actual URL of your deployed app)

## Features
- Upload a Pokémon TCG card image and receive detailed information about the card.
- Uses CNNs to predict the set symbol of the card.
- Applies OCR to extract the card’s collector number and total card count.
- Interactive web interface built with Streamlit.
- Supports identifying cards from multiple sets, with high accuracy.
- Allows users to confirm the identified card or attempt another search if the result is incorrect.

## Files and Directories
Here's an overview of the main files and directories used in the project:

- `app.py`: The main file to run the Streamlit app.
- `models/`: Directory containing pre-trained models for set symbol recognition.
- `ocr/`: Contains scripts for OCR preprocessing and extraction of collector numbers.
- `utils/`: Utility scripts used for image processing and data handling.
- `requirements.txt`: Contains a list of dependencies required to run the project.
- `README.md`: Documentation for the project.
- `data/`: (Optional) Directory for storing sample card images for testing purposes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
