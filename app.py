from flask import Flask, render_template, request, jsonify
from chat import get_response
from flask_cors import CORS


app = Flask(__name__)
"""allows website to be accessed by other domains"""
CORS(app)

"""renders an HTML Template base.html."""
@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

"""Is used for making predictions. send the input from input box to 'get_response' which return the result to the answer feild and return the answer to the user in json format"""
@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
