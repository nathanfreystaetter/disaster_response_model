# Disaster Response Model

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## Installation <a name="installation"></a>

In addition to the Anaconda distribution of Python, this notebook requires installation of SQLAlchemy to setup a sqllite database. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

My analysis was performed on data collected from social media platforms, and leveraged machine learning techniques to identify disasters and categorize for an appropriate and expedited response from an appropriate party.

## File Descriptions <a name="files"></a>

Data is stored in 2 CSV files located in the "Data" folder
Output from the analysis including a GUI has been deployed using Flask. Structure of the project includes

app
| | - template
| | - master.html # main page of web app
| | - go.html # classification result page of web app
| | - run.py # Flask file that runs app
data
| | - disaster_categories.csv # data to process
| | - disaster_messages.csv # data to process
| | - process_data.py
| | - InsertDatabaseName.db # database to save clean data to
models
| | - train_classifier.py
| | - classifier.pkl # saved model
README.md

## Results<a name="results"></a>

The main findings of my analysis can be found in the productionization of the Disaster Response App.
