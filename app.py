# Store this code in 'app.py' file
from flask import Response, stream_with_context
from flask import Response
from plyer import notification
from flask import Flask, render_template, request, redirect, url_for, session
import re
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import csv
import nltk
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Load the VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

face_classifier = cv2.CascadeClassifier(r'D:\Education Content-1\studi shan\B-Tech IT\3rd Year\6th Sem\Mini Project\Python\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\Education Content-1\studi shan\B-Tech IT\3rd Year\6th Sem\Mini Project\Python\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

app = Flask(__name__)
app.config['APP_NAME'] = 'ShanofRe'

app.secret_key = 'your secret key'
account = False
@app.route('/')
def index():
	return render_template('index.html')
@app.route('/login', methods =['GET', 'POST'])
def login():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		account = False
		username = request.form['username']
		password = request.form['password']
		with open ('register.csv', 'r') as file:
			reader = csv.reader(file)
			j = 0
			while j == 0:
				try:
					csv_row = ' '.join(next(reader))
				except StopIteration:
					j = 1
					break
				s = csv_row.split(" ")
				mail = s[2]
				ps = s[1]
				n = s[0]
				if username==mail and password ==ps:
					account = True
					break
		if account:
			session['loggedin'] = True
			session['username'] = n
			msg = 'Logged in successfully !'+n
			return render_template('vid.html', msg = msg)
		else:
			msg = 'Incorrect username / password !'
	return render_template('login.html', msg = msg)

@app.route('/logout')
def logout():
	session.pop('loggedin', None)
	session.pop('id', None)
	session.pop('username', None)
	return redirect(url_for('login'))

@app.route('/register', methods =['GET', 'POST'])
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		#check
		# if account:
		# 	msg = 'Account already exists !'
		if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers !'
		elif not username or not password or not email:
			msg = 'Please fill out the form !'
		else:
			row_list = [[username, password, email]]

			with open('register.csv', 'a', newline='') as file:
			    writer = csv.writer(file)
			    writer.writerows(row_list) 
			msg = 'You have successfully registered !'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg = msg)


@app.route('/video')
def video():
	return render_template('video.html')

def generate_frames():
	cap = cv2.VideoCapture(0)

	while True:
		emotion_counter = 0
		ret, frame = cap.read()
		if not ret:
			break
		labels = []
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
			roi_gray = gray[y:y+h,x:x+w]
			roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

			if np.sum([roi_gray])!=0:
				roi = roi_gray.astype('float')/255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi,axis=0)

				prediction = classifier.predict(roi)[0]
				label=emotion_labels[prediction.argmax()]
				current_emotion = label 
				with open('emotion.csv','a',newline='') as file:
					writer = csv.writer(file)
					writer.writerow([current_emotion])

				column_index = 0  # Update with the desired column index (0 for the first column, 1 for the second, and so on)

				# Read the CSV file
				with open('emotion.csv', 'r') as file:
				    reader = csv.reader(file)
				    data = list(reader)

				# Extract the last five elements from the specified column
				last_five_elements = [row[column_index] for row in data[-20:]]

				#print(last_five_elements)
				# Check if all elements are the same
				#print(len(set(last_five_elements)))
				if len(set(last_five_elements)) == 1:
					text =last_five_elements[0]
					cap.release()
					cv2.destroyAllWindows()
					print("Recommendation System Called")
					recom(text)
					return render_template('vid.html')		
				else:
					print("Detecting......")
				label_position = (x,y)
				cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
		# Convert the frame to JPEG format
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def recom(text):
	# Define a function to get the user's emotion
	def get_emotion(text):
	    # Use VADER to get the sentiment scores
	    sentiment = sia.polarity_scores(text)
	    # Determine the user's emotion based on the sentiment scores
	    if sentiment['compound'] >= 0.05:
	        return 'positive'
	    elif sentiment['compound'] <= -0.05:
	        return 'negative'
	    else:
	        return 'neutral'

	# Define some sample data
	data1 = {
	    'positive': ['You are greate my dear friend',"Believe in yourself; you are capable of achieving greatness.",
					    "Dream big, work hard, and turn your dreams into reality.",
					    "Success is the result of perseverance and a positive mindset.",
					    "Stay optimistic and focus on the possibilities ahead.",
					    "Your attitude determines your success; choose positivity.",
					    "Celebrate progress, no matter how small; it leads to big achievements.",
					    "Embrace challenges as opportunities for growth and self-improvement.",
					    "You have the power to overcome any obstacle and reach new heights.",
					    "Strive for excellence, and success will follow.",
					    "You are the architect of your own future; design it with determination and positivity.",
					    'You are always in postive', 'you have wonderful time'],
	    'negative': ['You are a biggest one dont worry about anything my dear friend',"Don't let failure define you; let it motivate you to try again.",
					    "In every difficulty lies an opportunity for growth and improvement.",
					    "Challenges are stepping stones to success; embrace them.",
					    "Mistakes are lessons that help you grow and become better.",
					    "Adversity reveals your true strength; keep pushing forward.",
					    "Turn setbacks into comebacks; let them fuel your determination.",
					    "Difficulties are temporary; your resilience is permanent.",
					    "Overcoming obstacles makes victory even sweeter.",
					    "Stay positive even when faced with negativity; it's your superpower.",
					    "In the face of challenges, you have the power to rise and shine.",
					    'This is not your last day kindly move on my dear friend', 
					    'I had a terrible experience'],
	    'neutral': ['This chair is comfortable', 'The weather is nice today',
					"Every day is a chance to learn and grow.",
				    "Embrace the journey and enjoy the process of self-discovery.",
				    "Stay open-minded and embrace new perspectives.",
				    "Find balance in all aspects of life; it leads to fulfillment.",
				    "Curiosity fuels learning and personal development.",
				    "Trust the journey and have faith in your abilities.",
				    "Draw inspiration from the beauty and wonders of the world.",
				    "Stay grounded in the present moment and appreciate the little things.",
				    "Find motivation in the support and encouragement of others.",
				    "Seek meaning and purpose in everything you do.",
	    			'Hi my dear anything secial today']
	}

	# Define a function to recommend items based on the user's emotion
	def recommend_item(emotion):
	    # Get the list of items for the user's emotion
	    items = data1[emotion]
	    # Return a random item from the list
	    return random.choice(items)


	# Get the user's emotion based on their input
	emotion = get_emotion(text)
	# Recommend an item based on the user's emotion
	item = recommend_item(emotion)
	msg = (f'Based on your emotion ({emotion}), we recommend: {item}')
	notification.notify(
        title='ShanofRe',
        message=msg,
        app_icon=None,  # Add path to your custom icon if desired
        timeout=10  # Duration to display the notification in seconds
	)
	return redirect(url_for('video'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=500, debug=True, threaded=False)