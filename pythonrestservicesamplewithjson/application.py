from flask import Flask
from flask import request, jsonify


app = Flask(__name__)


@app.route('/show', methods=['POST'])
def run():
    name = request.json["name"]
    age = request.json["age"]
    address = request.json["address"]

    print("name :", name)
    print("age :", age)
    print("address :", address)

    age = age + 5

    response = {
        "name": str(name),
        "age": age,
        "address": str(address)
    }
    return jsonify(response)
