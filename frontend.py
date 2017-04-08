import os
import captcha_cnn
from flask import Flask, render_template, jsonify, request
app = Flask(__name__)

@app.route("/")
def captcha():
    return render_template("playground.html")


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
