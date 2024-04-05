# VAE and RF Models Web Application

This project hosts a web application designed to interact with two distinct models: a Variational Autoencoder (VAE) and a Random Forest (RF) classifier. The application allows users to input data and receive predictions or generated samples from these models.

## Getting Started
These instructions will help you get a copy of the project running on your local machine for development and testing purposes.

### Installation

### Prerequisites
Python 3.x
pip (Python package manager)

### 1. **Clone the Project Repository**
        #Begin by cloning the repository to your local machine using the command:

        git clone https://github.com/srivatsan1303/CIMSEPP.git
        cd CIMSEPP

### 2. **Install Dependencies**
        #Install the necessary Python packages listed in `requirements.txt`:

        pip install -r requirements.txt

### 3. **Run the Application**
        #Execute main.py to start the web application::

        python main.py

## Usage

Upon running, the script will initialize a local server (typically Flask for Python applications), through which you can interact with the VAE and RF models.

Access the web application by navigating to http://localhost:5000 (or the port specified by your main.py script) in your web browser.

## Interacting with the Models
VAE Model: Provides an interface for generating new data samples based on input parameters.
RF Model: Offers prediction capabilities, classifying input data based on trained patterns.
Detailed usage instructions and options for interacting with these models will be provided within the web application's interface.

3# Development
Details about extending the application, such as adding new models or enhancing the UI, should be documented here for developers.
