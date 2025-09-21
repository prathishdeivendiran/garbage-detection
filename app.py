import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from twilio.rest import Client

# Load YOLOv8 model (your best.pt)
model = YOLO("best.pt")

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# WhatsApp setup (Twilio example)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_WHATSAPP = "whatsapp:+14155238886"   # Twilio sandbox number
ADMIN_WHATSAPP = "whatsapp:+91XXXXXXXXXX"   # Replace with your number
client = Client(TWILIO_SID, TWILIO_AUTH)

def send_whatsapp_message(message):
    client.messages.create(
        from_=TWILIO_WHATSAPP,
        body=message,
        to=ADMIN_WHATSAPP
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded video
        file = request.files["video"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Run YOLO inference
        results = model.predict(source=filepath, save=True, project="runs", name="inference")
        status = "Not Filled"

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "filled":
                    status = "Filled"
                    send_whatsapp_message("⚠️ Garbage bin is filled! Please take action.")
                    break

        return render_template("result.html", status=status, video=file.filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
