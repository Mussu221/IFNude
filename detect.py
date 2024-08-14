from flask import Flask, render_template, request, jsonify
from flask_restful import Api, Resource
import tempfile
from flask_cors import CORS
from ifnude import detect
import os

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

class NudityCheck(Resource):
    def post(self):
        temp_file_path = None
        try:
            # Check if file is present in the request
            if 'file' not in request.files:
                return {"status": 0, "message": "File is required"}, 400

            file = request.files.get('file')
            mode = request.form.get("mode", "default")
            threshold = request.form.get("threshold", "50")

            # Validate mode
            if mode not in ["default", "fast"]:
                return {"status": 0, "message": "Mode should be one of ['default', 'fast']"}, 400

            # Validate threshold
            try:
                threshold = float(threshold)
                if not (0.1 <= threshold <= 1.0):
                    return {"status": 0, "message": "Threshold should be between 0.1 to 1.0"}, 400
            except ValueError:
                return {"status": 0, "message": "Threshold should be a valid integer between 0.1 to 1.0"}, 400

            # Validate file
            if file.filename == '':
                return {"status": 0, "message": "No file selected for uploading"}, 400

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            # Check for nudity
            score = detect(temp_file_path, mode=mode, min_prob=threshold)

            return {"status": 1, "message": "Image processed successfully", "data": {"has_nudity": True if score else False, "score":score}}, 200

        except Exception as e:
            print(str(e))
            return {"status": 0, "message": "An error occurred during processing. Please try again later."}, 500

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("detect.html")

# Add the route to the API


# Add the NudityCheck resource to the API
api.add_resource(NudityCheck, '/check_nudity')

if __name__ == '__main__':
    app.run(port=1016, debug=True, host="0.0.0.0")
