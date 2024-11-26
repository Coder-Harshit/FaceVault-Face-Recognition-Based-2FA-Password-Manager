import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import threading
import json
from cryptography.fernet import Fernet
import getpass
import mysql.connector
import bcrypt

# Global variables
user_input = ''
input_lock = threading.Lock()

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="",  
        database="user_db", port= 3307
    )

def store_password(username, password):
    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    db_connection = connect_db()
    cursor = db_connection.cursor()

    try:
        # Insert the new user and their encrypted password into the database
        cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""")
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password.decode('utf-8')))
        db_connection.commit()
        print(f"✅ Password for user '{username}' stored securely in the database.")
    except mysql.connector.errors.IntegrityError:
        print(f"❌ User with the username'{username}' already exists. Please try a different username")
        return "false" #added here
    except Exception as e:
        print(f"❌ Error storing password: {e}")
    finally:
        cursor.close()
        db_connection.close()

def verify_password(username, password):
    db_connection = connect_db()
    cursor = db_connection.cursor()

    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    stored_password = cursor.fetchone()

    if stored_password:
        if bcrypt.checkpw(password.encode('utf-8'), stored_password[0].encode('utf-8')):
            print(f"✅ Password match for user '{username}'.")
            return True
        else:
            print(f"❌ Incorrect password for user '{username}'.")
            return False
    else:
        print(f"❌ User '{username}' not found.")
        return False

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    print("\nEncoding saved faces...Please Wait")
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])
               
            except IndexError:
                print(f" Could not find a face in {filename}. Skipping.")

    print("\n All faces encoded successfully.")
    return known_face_encodings, known_face_names

def generate_key():
    """Generate a new encryption key and save it to a file."""
    key = Fernet.generate_key()
    with open("encryption.key", "wb") as f:
        f.write(key)
    print(" New encryption key generated and saved.")
    return key

def fetch_key():
    """Fetch the encryption key from the file."""
    key_file = "encryption.key"
    if not os.path.exists(key_file):
        print("❌ Encryption key not found. Generating a new one...")
        return generate_key()
    with open(key_file, "rb") as f:
        return f.read()
    
def encrypt_data(data, key):
    """Encrypt the data using Fernet symmetric encryption."""
    f = Fernet(key)
    json_str = json.dumps(data)
    encrypted_data = f.encrypt(json_str.encode())
    return encrypted_data

def decrypt_data(encrypted_data, key):
    """Decrypt the data using Fernet symmetric encryption."""
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return json.loads(decrypted_data.decode())

def load_user_data(file_path):
    """Load user data, handling both encrypted and unencrypted formats."""
    try:
        key = fetch_key()  #changed from generate key
        
        if not os.path.exists(file_path):
            print(f"ℹ️ No existing user data file found. Creating new one.")
            return {}
            
        with open(file_path, 'r') as file:
            try:
                # First try to load as unencrypted JSON
                user_data = json.load(file)
                print("ℹ️ Found unencrypted data. Converting to encrypted format...")
                # Convert to encrypted format
                save_user_data(file_path, user_data)
                return user_data
            except json.JSONDecodeError:
                # If that fails, try to load as encrypted data
                file.seek(0)
                encrypted_data = file.read().encode()
                try:
                    user_data = decrypt_data(encrypted_data, key)
                    print("✅ Encrypted user data loaded successfully.")
                    return user_data
                except Exception as e:
                    print(f"❌ Error decrypting data: {e}")
                    return {}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {}
                
def display_user_info(user_data, name):
    """Display decrypted user information."""
    if name in user_data:
        user_info = user_data[name]
        print("\n Decrypted User Information:")
        print(f"👤 Name: {name.capitalize()}")
        for k, val in user_info.items():
            print(f"\n {k.capitalize()}: {val}")
    else:
        print(f" No information found for {name}")

def fetch_one(user_data, name, site):
    """Fetch and display specific site information for a user."""
    if name in user_data:
        user_info = user_data[name]
        if site in user_info:
            print(f"{site.capitalize()}: {user_info[site]}")
        else:
            print(f" Password for {site} not found.")
    else:
        print(f" No information found for user '{name}'.")

def save_user_data(file_path, user_data):
    """Encrypt and save user data to file."""
    try:
        key = fetch_key()
        encrypted_data = encrypt_data(user_data, key)
        
        with open(file_path, 'wb') as file:
            file.write(encrypted_data)
        print(" User data encrypted and saved successfully.")
    except Exception as e:
        print(f" Error saving encrypted user data: {e}")

def capture_input():
    global user_input
    while True:
        with input_lock:
            user_input = input("Enter 'u' to add new user, 's' to capture or 'q' to quit ")
            if user_input.lower() in ['s', 'q']:
                break
                                
def capture_face_with_rectangle(frame, face_classifier):
    """Detect faces and draw rectangles on the frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    # Draw rectangles around the faces and return the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame, faces

def detect_and_recognize_faces(frame, known_face_encodings, known_face_names, face_classifier, previous_names, user_data):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    recognized_names = []
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        recognized_names.append(name)

    # Check for newly recognized faces
    new_recognitions = set(recognized_names) - set(previous_names)
    if new_recognitions:
        for name in new_recognitions:
            if name != "Unknown":
                print(f"\nFace recognized: {name.capitalize()}")
             #   display_user_info(user_data, name)    changed here

    return frame, recognized_names


def add_new_user(known_face_encodings, known_face_names, user_data, directory):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not open video capture.")
        return

    print("\n📸 Capturing new user's face. Please position your face inside the green rectangle.")
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture image.")
            cap.release()
            return

        processed_frame, faces = capture_face_with_rectangle(frame, face_classifier)
        cv2.imshow("Face Capture", processed_frame)

        if len(faces) > 0:  # If a face is detected
            print("✅ Face detected. Press 'c' to capture the image or 'q' to quit.")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Capture the image when 'c' is pressed
                # Capture face encoding
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                if face_encodings:
                    new_face_encoding = face_encodings[0]
                else:
                    print("Failed to encode face. Please try again.")
                    continue

                # Validate name only
                while True:
                    name = input("Enter the user's name: ").strip()
                    if name and not any(char in name for char in ['/', '\\', '*', '?', ':', '|', '<', '>', '"']):
                        break
                    print(" Invalid name. Please avoid special characters.")
                password = input("Enter the user's password: ").strip()

                fl=store_password(name.lower(),password)
                if fl=="false":
                    return "false"

                # Collect additional user information without validation
                flag = True
                new_user_data={}
                while(flag):
                    print("\nDo you want to enter more data?")
                    inp= input("Reply: ")
                    if inp.lower()=="y":
                        site=input("Enter site: ")
                        pwd = getpass.getpass("Enter password: ")
                        new_user_data[site.lower()] = pwd
                    elif inp.lower()=="n":
                        flag=False
                    else:
                        print(" Invalid input. Please reply with 'y' or 'n'.")

                try:
                    # Save the captured image
                    image_path = os.path.join(directory, f"{name}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f" Image saved as '{image_path}'")

                    # Save face encoding
                    encoding_path = os.path.join(directory, f"{name}.npy")
                    np.save(encoding_path, new_face_encoding)
                    known_face_encodings.append(new_face_encoding)
                    known_face_names.append(name)

                    # Update the user_data dictionary
                    user_data[name] = new_user_data

                    # Encrypt and save data
                    key = fetch_key()
                    encrypted_data = encrypt_data(user_data, key)

                    # Save encrypted data to file
                    with open("user_data.json", 'wb') as file:
                        file.write(encrypted_data)

                    print(f" New user '{name}' added successfully with encrypted data.")
                    
                    # Display confirmation of stored data
                    print("\n🔒 Encrypted user information stored:")
                    print(f"👤 Name: {name}")
                    for site, password in new_user_data.items():
                         print(f"🔗 Site: {site.capitalize()}")
                         print(f"🔑 Password: {password}")
                    
                    break 

                except Exception as e:
                    print(f" Error saving user data: {str(e)}")
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    if os.path.exists(encoding_path):
                        os.remove(encoding_path)
                    print("⚠️ Changes were rolled back due to error.")
                    continue

            elif key == ord('q'):  # Quit if 'q' is pressed
                print(" Exiting capture mode.")
                break

        else:
            print("⚠️ No face detected. Try again.")

def addPassword(user_data, name):
    site = input("Enter a site name: ").strip().lower()
    
    if site in user_data[name]:
        inp= input("Password already exists. Do you want to add a new one? ")
        if inp.lower()=='y':
            pwd = getpass.getpass("Enter password: ")
            if name in user_data:
                # Ensure the user's data contains a dictionary for sites and passwords
                if isinstance(user_data[name], dict):
                    user_data[name][site] = pwd
                    print(f"✅ Password for '{site.capitalize()}' added for user '{name}'")
                else:
                    print("❌ Error: User data is not properly formatted.")
                   
    else:
                # If the name doesn't exist in the data, create a new entry
        pwd = getpass.getpass("Enter password: ")
        if isinstance(user_data[name], dict):
                    user_data[name][site] = pwd
                    print(f"✅ Password for '{site.capitalize()}' added for user '{name}'")
        else:
                    print("❌ Error: User data is not properly formatted.")
    return


def deletePassword(user_data, name):
    site = input("Enter a site name to delete the password for: ").strip().lower()

    if name in user_data:
        # Check if the user's data is a dictionary and contains the site
        if isinstance(user_data[name], dict):
            if site in user_data[name]:
                # Delete the password for the site
                user_data[name].pop(site, None)
                print(f"✅ Password for '{site}' has been deleted for user '{name}'.")
            else:
                print(f"❌ No password found for '{site}' under user '{name}'.")
        else:
            print(f"❌ Error: User '{name}' data is not a dictionary of sites and passwords.")
    else:
        print(f"❌ User '{name}' not found in the data.")

    return


def main():
    global user_input
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video capture.")
        return

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    known_face_encodings, known_face_names = load_known_faces("saved_faces")
    user_data = load_user_data("user_data.json")

    print("\n📸 Starting video capture. Press 'u' to add user, 's' to capture and retrive info or 'q' to quit.")
    previous_names = []

    input_thread = threading.Thread(target=capture_input)
    input_thread.daemon = True
    input_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        processed_frame, recognized_names = detect_and_recognize_faces(
            frame, known_face_encodings, known_face_names, face_classifier, previous_names, user_data
        )
        previous_names = recognized_names

        cv2.imshow('Face Recognition System', processed_frame)

        with input_lock:
            if user_input == 'u':
                flag=add_new_user(known_face_encodings, known_face_names, user_data, "saved_faces")
                user_input = ''
                if flag=="false":
                    print("User not Added")
                    break
                else:
                    print("New user added successfully. Resuming video capture.")
                continue
            
            if user_input == 's':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                captured_image_path = f"captured_{timestamp}.jpg"
                cv2.imwrite(captured_image_path, frame)
                print(f"\n📸 Image captured and saved as '{captured_image_path}'")

                cap.release()
                cv2.destroyAllWindows()

                captured_image = cv2.imread(captured_image_path)
                captured_encodings = face_recognition.face_encodings(captured_image)

                if captured_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, captured_encodings[0])
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_face_encodings, captured_encodings[0])
                        best_match_index = np.argmin(face_distances)
                        matched_name = known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        print(f"✅ Match found! {matched_name} (Confidence: {confidence:.2%})")
                        password=input("Enter your password: ")
                        if (verify_password(matched_name, password)):
                            print("Do you want all information or a specific (a/s)? ")
                            inp= input("Reply: ")
                            if inp=='a':
                                display_user_info(user_data, matched_name)
                            elif inp=='s':
                                site= input("Enter a Site name: ")
                                fetch_one(user_data, matched_name , site.lower())
                            
                            print("Do you want to add a new password? (y/n) ")
                            inp = input("Reply: ").strip().lower()
                                
                            if inp == 'y':
                                    key=fetch_key()
                                    addPassword(user_data, matched_name)
                                    try:
                                        encrypted_data = encrypt_data(user_data, key)
                                        with open("user_data.json", "wb") as file:
                                                file.write(encrypted_data)
        
                                    except Exception as e:
                                        print(f"❌ Error saving data: {e}")
                            

                            print("Do you want to delete a  password? (y/n) ")
                            inp = input("Reply: ").strip().lower()

                            if inp == 'y':
                                    key=fetch_key()
                                    deletePassword(user_data, matched_name)
                                    try:
                                        encrypted_data = encrypt_data(user_data, key)

                                        with open("user_data.json", "wb") as file:
                                            file.write(encrypted_data)

                                        print(f"✅ Password deleted and data updated successfully for user '{matched_name}'.")
                                    except Exception as e:
                                        print(f"❌ Error saving data: {e}")
                            
        
                        else:
                            print("No match found.")
                        
                    else:
                        print(" No face detected.")

                    continue

            elif user_input == 'q':
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs("saved_faces", exist_ok=True)
    main()