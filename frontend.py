import os
import captcha_cnn
import video_classifier
from flask import Flask, render_template, jsonify, request, flash
from flask import redirect, url_for
from flask import send_from_directory
import uuid
from werkzeug.utils import secure_filename
from werkzeug import SharedDataMiddleware
import skvideo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "upload/"
app.config['SECRET_KEY'] = "upload/"
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
        '/upload':  app.config['UPLOAD_FOLDER']
})

@app.route("/")
def captcha():
    return render_template("playground.html")

@app.route("/video/", methods=["POST", "GET"])
def video():
    if request.method == "POST":
        if 'video' not in request.files:
            flash("No file apart")
            return redirect(request.url)
        video = request.files["video"]
        if video and video.filename.endswith("mp4"):
            filename = secure_filename("{}.mp4".format(uuid.uuid4()))
            video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result', filename=filename))
    return render_template("video.html")

@app.route("/result/")
def result():
    filename = request.args.get("filename")
    video = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    stats = video_classifier.analyze_video(str(video), .1)
    summary = video_classifier.get_summary(stats)
    metadata = skvideo.io.ffprobe(str(video))
    exec("frame_rate = {}".format(metadata["video"]["@r_frame_rate"]))
    return render_template("result.html", summary=summary, video=video, frame_rate=frame_rate)


@app.route("/gencaptcha")
def gen_captcha():
    if not os.path.exists("static"):
        os.mkdir("static")
    path = captcha_cnn._gen_one_captcha("static")
    return jsonify(path=path)

@app.route("/recognize")
def recognize():
    model = captcha_cnn.load_trained_model()
    path = request.args.get("path")
    result = captcha_cnn.predict(model, path)
    real = path.split("/")[-1].split("-")[0]
    return jsonify(real=real, result=result)
    

if __name__ == "__main__":
    app.run(debug=True)
