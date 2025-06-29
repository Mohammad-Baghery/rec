Advanced Face Recognition & Identity Management System
Overview
This project is a powerful, modular backend engine designed for a wide range of face recognition and identity management tasks. It features a decoupled, data-centric architecture where all operations are performed on database entities rather than direct file paths.

Communication with the front-end is handled through a modern event-driven system using Blinker signals, and data persistence is managed by MongoDB, ensuring scalability and flexibility. The system is optimized to work with pre-cropped face images for maximum efficiency.

Key Features
The application operates in different modes, each designed for a specific data-processing task:

Data Ingestion & Indexing
add_faces: A UI-simulated task that adds new face image file paths to the database for future processing. This is the primary entry point for new data.

index: A core backend task that processes all un-indexed faces in the database by calculating and storing their feature embeddings.

Identity Management
enroll: Enrolls a new person with a specific name by associating a list of pre-indexed face IDs with them.

identify: Identifies a person in a single photo (referenced by its database ID) by comparing it against all known persons.

find_duplicates: Compares all enrolled persons against each other to find pairs that are likely the same individual.

merge_persons: Merges two person profiles, consolidating all faces under one identity and deleting the duplicate.

Batch Operations & Analysis
sort: Clusters unassigned faces (based on a list of face IDs) and creates new, distinct person profiles for each discovered group.

compare_one_vs_many: Compares a single specific face against a list of other faces using their database IDs.

batch_compare: Compares a group of faces against another group using their database IDs.

Tech Stack
Database: MongoDB for flexible and scalable storage of person and face data.

Event System: Blinker for clean, decoupled communication between the backend engine and the user interface.

Face Processing: PyTorch & FaceNet-PyTorch for robust facial feature embedding extraction.

Clustering: Scikit-learn (DBSCAN) for intelligent grouping of similar faces.

Prerequisites
Before running the application, ensure you have the following installed and running:

Python 3.8 or newer.

MongoDB Community Server: The service must be installed and running on your system.

pip: Python's package installer.

Installation and Setup
Create a Virtual Environment:

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install Required Libraries:
Create a requirements.txt file in your project root with the following content:

torch
torchvision
torchaudio
pymongo
blinker
dnspython
numpy
opencv-python
facenet-pytorch
scikit-learn
matplotlib
tqdm

Then, install them using pip:

pip install -r requirements.txt

Configure Database Connection:
Open the config.py file and ensure the MONGO_URI variable points to your MongoDB server. For a standard local installation, the default value is usually correct:

MONGO_URI = "mongodb://localhost:27017/"

How to Run and Test
All operations can be managed and tested through the test_runner.py script, which acts as a central control panel.

General Steps:

Open the test_runner.py file.

In the USER CONTROL PANEL section at the top, set the SELECTED_MODE variable to the desired operation (e.g., "add_faces", "index", "sort").

Configure the relevant paths and parameters for that mode in the same section.

Run the script as a standard Python file from your terminal:

python test_runner.py

Example Workflow
A common workflow for processing a new batch of photos would be:

Add Face Paths to DB:

Set SELECTED_MODE = "add_faces".

Specify the path to your new photos in PATH_SOURCE_FOLDER.

Run the script. This simulates the UI adding new records to the faces collection.

Index New Faces:

Set SELECTED_MODE = "index".

Run the script. The backend will find all unprocessed faces, calculate their embeddings, and save them to the database.

Sort and Create Identities:

Set SELECTED_MODE = "sort".

Run the script. The backend will take all indexed but unassigned faces, cluster them, and create new Person documents for each group.

Identify a Photo:

Set SELECTED_MODE = "identify".

Set the path to the photo you want to identify in PATH_TO_IDENTIFY. The script will find its ID in the DB and run the identification.

Run the script to see the result.

Project Structure
/src: Contains all core application logic.

face_recognition_app.py: The main engine that orchestrates all operations.

face_db_manager.py: Handles all interactions with the MongoDB database.

feature_extractor.py: Responsible for extracting feature embeddings from images.

signals.py: Defines the Blinker signals for UI communication.

base_app.py: Base class for managing process state (active, status, etc.).

test_runner.py: The main script for running and testing different application modes.

config.py: Central configuration file for all global settings.

database/: This folder can hold the Annoy index file.

results/: The default output directory for any generated image or JSON results.
