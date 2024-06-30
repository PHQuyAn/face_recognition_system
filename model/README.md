### Train the Model (`train.py`) ✨  
Read images from the data_images directory and extract facial features using the FaceNet algorithm.  
After that, train the SVM model using the facial feature dataset.  

### Save to Database (`train_savedatabaseC.py`)  
Similar to train.py but instead of saving the model to a file, it stores facial feature information into a PostgreSQL database.  

### Live Camera Evaluation (`live_cam.py`)
Detect faces in the images, extract facial features, and use the SVM model to classify the user.  
Display the recognized user's name or a "Stranger" message if no match is found in the database.
