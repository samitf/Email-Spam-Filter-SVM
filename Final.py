

from flask import Flask, jsonify, render_template, request, redirect, session, url_for  #web framework

# Import modules
import imaplib #get mail
import smtplib #send mail
import email #handling mail
from email.header import decode_header #decode mails
import pandas as pd #for dataset
from sklearn.feature_extraction.text import CountVectorizer #svm model
from sklearn.svm import SVC
import re #match strings
from email.mime.text import MIMEText #mail format


app = Flask(__name__) #define flask name
app.secret_key = 'hrsz miqx fngf sbea' #set a key for current session

data = pd.read_csv('mail_data.csv', encoding='latin-1') #load the dataset and encode

# Convert labels to binary (0 for ham, 1 for spam)
data['label'] = pd.get_dummies(data['Category'])['spam']

# Vectorize the email text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X, data['label'])

# Dictionary of email providers and their SMTP server/port settings
email_providers = {
    'gmail.com': ('smtp.gmail.com', 587),
    'outlook.com': ('smtp.office365.com', 587),
    'hotmail.com': ('smtp.office365.com', 587),
    'yahoo.com': ('smtp.mail.yahoo.com', 587)
}


blacklist = ["Twitter", "@twitter.com", "Flipkart", "@flipkart.com", "Spotify", "@spotify.com","Valued Opinions"]

global username
global password

#login function
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        global username
        global password
        username = request.form['username']
        password = request.form['password']
        # get the email provider from the username
        provider = username.split('@')[1]
        # set the host and port according to the provider
        if provider == 'gmail.com':
            host = 'imap.gmail.com'
            port = 993
        elif provider == 'outlook.com':
            host = 'outlook.office365.com'
            port = 993
        elif provider == 'yahoo.com':
            host = 'imap.mail.yahoo.com'
            port = 993
        else:
            error = 'Unsupported email provider.'
            return render_template('login.html', error=error)
        try:
            imap = imaplib.IMAP4_SSL(host, port)
            imap.login(username, password)
            # if the email and password are correct, set the 'logged_in' session variable to True
            session['logged_in'] = True
            save_user_email()  # save the user email in the session object
            imap.logout()
            # redirect to the index page
            return redirect(url_for('index'))
        except imaplib.IMAP4.error:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

#save the user email of currently logged in email
def save_user_email():
    if session.get('logged_in'):
        session['user_email'] = username

#test page to check if login is successful
# this is a protected page that requires a user to be logged in
@app.route('/protected')
def protected():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        return "This is a protected page."

#used to get the accuracy and user email to frontend
@app.route('/')
def index():
    # Vectorize the email text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Message'])

    # Train the SVM model
    model = SVC(kernel='linear')
    model.fit(X, data['label'])
    accuracy = model.score(X, data['label'])  # calculate the accuracy of the model

    if session.get('logged_in'):
        user_email = session['user_email']
        return render_template('index.html', accuracy=accuracy, user_email=user_email)
    else:
        return render_template('index.html', accuracy=accuracy)

#get emails from currently logged in mail
@app.route('/api/get_emails')
def get_emails():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    Message_HAM = []
    Date_HAM = []
    Subject_HAM = []
    From_HAM = []
    Message_SPAM = []
    Date_SPAM = []
    Subject_SPAM = []
    From_SPAM = []
    Message_ALL = []

    # Vectorize the email text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Message'])

    # Train the SVM model
    model = SVC(kernel='linear')
    model.fit(X, data['label'])

    accuracy = model.score(X, data['label'])  # calculate the accuracy of the model

    # Connect to email server
    user = username
    passw = password
    # get the email provider from the username
    provider = user.split('@')[1]
    # set the host and port according to the provider
    if provider == 'gmail.com':
        host = 'imap.gmail.com'
        port = 993
    elif provider == 'outlook.com':
        host = 'outlook.office365.com'
        port = 993
    elif provider == 'yahoo.com':
        host = 'imap.mail.yahoo.com'
        port = 993
    imap = imaplib.IMAP4_SSL(host, port)
    imap.login(user, passw)
    # Select inbox folder by default
    status, messages = imap.select("INBOX")
    messages = int(messages[0])

    for i in range(messages, messages -20, -1): #Changing the index will determine the amount of mails that you will fetch e.g.-(messages - 50, -1) will fetch 50 latest mails from Inbox
        # Fetch the message
        res, msg = imap.fetch(str(i), "(RFC822)")
        # Parse the message
        msg = email.message_from_bytes(msg[0][1])
        # Decode the header fields
        From, encoding = decode_header(msg["From"])[0]
        if isinstance(From, bytes):
            From = From.decode(encoding or 'utf-8')
        Subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(Subject, bytes):
            Subject = Subject.decode(encoding or 'utf-8')
        Date = msg["Date"]
        # Get the plain text content of the message
        Message = get_body(msg)
        Message = re.sub('<[^>]+>', '', Message)
        # Wrap the long text into multiple lines
        # Message = "\n".join(wrap(Message, width=200))
        # Vectorize the input text
        input_vector = vectorizer.transform([Message])
        # Make a prediction
        prediction = model.predict(input_vector)[0]
        if prediction == 1 or any(x in From for x in blacklist):
            Message_SPAM.append(Message)
            Date_SPAM.append(Date)
            Subject_SPAM.append(Subject)
            From_SPAM.append(From)
        else:
            Message_HAM.append(Message)
            Date_HAM.append(Date)
            Subject_HAM.append(Subject)
            From_HAM.append(From)
        Message_ALL.append(
            {"Label": "spam" if prediction == 1 else "ham", "From": From, "Subject": Subject, "Date": Date,
             "Message": Message})

    imap.logout()
    # Return JSON response
    response = {
        "HAM_MESSAGES": [{"From": From_HAM[i],
                          "Subject": Subject_HAM[i],
                          "Date": Date_HAM[i],
                          "Message": Message_HAM[i]
                          } for i in range(len(From_HAM))
                         ],
        "SPAM_MESSAGES": [{"From": From_SPAM[i],
                           "Subject": Subject_SPAM[i],
                           "Date": Date_SPAM[i],
                           "Message": Message_SPAM[i]
                           } for i in range(len(From_SPAM))
                          ],
        "ALL_MESSAGES": Message_ALL,
        "ACCURACY": accuracy  # include the accuracy in the response

    }

    return jsonify(response)

#send_mail through currently logged in user
@app.route('/api/send_email', methods=['POST'])
def send_email():
    user = username
    passw = password

    # Get request data
    data = request.get_json()

    # Validate input data
    if not all(key in data for key in ['from', 'to', 'subject', 'message']):
        return jsonify({'message': 'Missing required fields'}), 400

    # Validate 'from' field
    sender = data['from']
    if not re.match(r"[^@]+@[^@]+\.[^@]+", sender):
        return jsonify({'message': 'Invalid sender email address'}), 400

    # Determine SMTP server and port based on email provider
    provider = sender.split('@')[1]
    if provider not in email_providers:
        return jsonify({'message': 'Unsupported email provider'}), 400
    smtp_server, smtp_port = email_providers[provider]

    # Set up email message
    msg = MIMEText(data['message'])
    msg['From'] = sender
    msg['To'] = data['to']
    msg['Subject'] = data['subject']

    # Connect to SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(user, passw)

    # Send email and close connection
    server.sendmail(sender, data['to'], msg.as_string())
    server.quit()

    return jsonify({'message': 'Email sent successfully'})

#decode the mails fetched
def get_body(msg):
    if msg.is_multipart():
        return get_body(msg.get_payload(0))
    else:
        return msg.get_payload(None, True).decode('latin-1')

@app.route('/password')
def password():
    return render_template ('password.html')

if __name__ == '__main__':
    app.run(debug=True)

# http://localhost:5000/api/get_emails    -API
# http://127.0.0.1:5000/login      -Login page
