# *FaceVault: Face Recognition-Based 2FA Password Manager*

Securely manage your passwords with a modern face recognition-based login and two-factor authentication (2FA).

## About the Project

FaceVault is an innovative face recognition-based 2FA password manager designed for security and ease of use. It leverages facial recognition for login authentication, bcrypt for securely hashing passwords, and Fernet encryption for local storage of sensitive data.
	
•	Secure Login: Combines face recognition with hashed passwords for robust 2FA.

•	Safe Credentials Storage: Passwords and user data are encrypted and stored locally in user_data.json.

•	Manage Users and Passwords: Add new users, securely store credentials, and retrieve them when needed.

## Usage
1.	Add a New User
   
	•	Step 1: Position your face in the frame to capture your facial encoding.

	•	Step 2: Enter a name and a password (hashed using bcrypt).

	•	Step 3: Optionally store additional site credentials.

2.	Authenticate a User
   
	•	Step 1: Face recognition matches your face with stored encodings.

	•	Step 2: Enter your password to complete 2FA.

3.	Retrieve or Manage Credentials
   
	•	Fetch all stored credentials or add new ones dynamically.

## Libraries Used
•	Python

•	OpenCV: Face detection with Haar cascades.

•	Face Recognition: Facial encoding and matching for authentication.

•	Cryptography (Fernet): AES-based encryption for secure data storage.

•	Bcrypt: Salting and hashing for password security.

•	MySQL Connector: For database operations (requires configuration).

## Notes
•	Ensure that you configure your MySQL connector ports, database name and table name according to your system.

•	The project expects a saved_faces folder containing images of the saved users. This is where the facial encodings are stored and matched.
