VAE and RF Models Web Application
This project hosts a web application designed to interact with two distinct models: a Variational Autoencoder (VAE) and a Random Forest (RF) classifier. The application allows users to input data and receive predictions or generated samples from these models.

Getting Started
These instructions will help you get a copy of the project running on your local machine for development and testing purposes.

Prerequisites
Python 3.x
pip (Python package manager)
Setup
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/srivatsan1303/CIMSEPP.git
cd CIMSEPP
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Running the Application
Execute main.py to start the web application:

bash
Copy code
python main.py
Upon running, the script will initialize a local server (typically Flask for Python applications), through which you can interact with the VAE and RF models.

Access the web application by navigating to http://localhost:5000 (or the port specified by your main.py script) in your web browser.

Interacting with the Models
VAE Model: Provides an interface for generating new data samples based on input parameters.
RF Model: Offers prediction capabilities, classifying input data based on trained patterns.
Detailed usage instructions and options for interacting with these models will be provided within the web application's interface.

Development
Details about extending the application, such as adding new models or enhancing the UI, should be documented here for developers.
