import os
import zlib
import base64
import subprocess
import face_recognition  # type: ignore
from cs50 import SQL  # type: ignore
from tempfile import mkdtemp
from base64 import b64decode
from flask_session import Session  # type: ignore
from helpers import login_required
from werkzeug.security import check_password_hash, generate_password_hash  # type: ignore
from flask import Flask, flash, redirect, render_template, request, session  # type: ignore
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError  # type: ignore


# Configure application
app = Flask(__name__)
#configure flask-socketio

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///data.db")


@app.route("/")
@login_required
def home():
    return redirect("/home")


@app.route("/home")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()

    if request.method == "POST":

        input_username = request.form.get("username")
        input_password = request.form.get("password")

        if not input_username:
            return render_template("login.html",messager = 1)

        elif not input_password:
             return render_template("login.html",messager = 2)

        # Query database for username
        username = db.execute("SELECT * FROM users WHERE username = :username",
                              username=input_username)

        # Ensure correct identification
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        session["user_id"] = username[0]["id"] # -> Remember user...

        config_dir = os.getcwd()
        command = "chainlit run model.py"  
        subprocess.run(command, shell=True, cwd=config_dir.replace('auth', ''))
        return redirect("/")

    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        if not input_username:
            return render_template("register.html",messager = 1)

        elif not input_password:
            return render_template("register.html",messager = 2)

        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        username = db.execute("SELECT username FROM users WHERE username = :username",
                              username=input_username)

        if len(username) == 1: # -> make sure username not already taken
            return render_template("register.html",messager = 5)

        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                session["user_id"] = new_user

            flash(f"Registered as {input_username}")

            config_dir = os.getcwd() 
            command = "chainlit run model.py"  
            subprocess.run(command, shell=True, cwd=config_dir.replace('auth', ''))
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":
        encoded_image = (request.form.get("pic")+"==").encode('utf-8')

        decoded_data = base64.b64decode(encoded_image)
        temp_image_path = os.path.join('static', 'face', '1.jpg')
        
        with open(temp_image_path, 'wb') as temp_image_handle:
            temp_image_handle.write(decoded_data)

        face_folder = os.path.join('static', 'face')
        image_files = [file for file in os.listdir(face_folder) if file.endswith('.jpg')]

        scanned_image = face_recognition.load_image_file(temp_image_path)
        scanned_face_encodings = face_recognition.face_encodings(scanned_image)

        if not scanned_face_encodings:
            return render_template("camera.html", message=2)  # No face detected in the scanned image
        
        scanned_face_encoding = scanned_face_encodings[0]

        for image_file in image_files:
            known_image_path = os.path.join('static', 'face', image_file)
            known_image = face_recognition.load_image_file(known_image_path)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]

            results = face_recognition.compare_faces([known_face_encoding], scanned_face_encoding)

            if results[0]:  
                username = db.execute("SELECT * FROM users WHERE username = :username",
                                username="swa")
                session["user_id"] = username[0]["id"]
                print("Access granted")

                config_dir = os.getcwd()
                command = "chainlit run model.py"  
                subprocess.run(command, shell=True, cwd=config_dir.replace('auth', ''))
                return redirect("/")
            
        else:
            return render_template("camera.html", message=3)  # Incorrect face
        
    else:
        return render_template("camera.html")


@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":
        encoded_image = (request.form.get("pic")+"==").encode('utf-8')

        id_=db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]
        # id_ = db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_bill = face_recognition.load_image_file(
        './static/face/'+str(id_)+'.jpg')    
        try:
            bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
        except:    
            return render_template("face.html",message = 1)
        return redirect("/home")

    else:
        return render_template("face.html")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
      app.run()
