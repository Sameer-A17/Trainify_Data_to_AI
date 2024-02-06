from preprocessing import supervised_preprocessing, generic_clustering
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, url_for, send_file

app = Flask(__name__)


def find_model_file(model_name):
    root_dir = 'Models'
    for model_type in os.listdir(root_dir):
        model_type_dir = os.path.join(root_dir, model_type)
        model_file_path = os.path.join(model_type_dir, f"{model_name}.py")
        if os.path.isfile(model_file_path):
            return model_file_path
    return None


@app.route('/download_python_file/<model_name>', methods=["GET"])
def download_python_file(model_name):
    try:
        file_path = find_model_file(model_name)

        if file_path is None:
            return "File not found", 404

        return send_file(file_path, as_attachment=True, download_name=f"{model_name}.py")
    except Exception as e:
        return jsonify({"error": str(e)}), 400





@app.route('/predict/<Model_Type>', methods=['POST'])
def predict(Model_Type):
    try:
        loaded_model = joblib.load('Best_Model.pkl')
        data = request.files['file']
        df = pd.read_csv(data)
        df_cleaned = df.dropna(axis=1)
        predictions = loaded_model.predict(df_cleaned)

        if Model_Type == "Classification":
            with open('original_labels.txt', 'r') as file:
                original_labels = [label.strip() for label in file.readlines()]

            predictions = [original_labels[prediction] for prediction in predictions]

            response = {
                "Predictions": predictions
            }

            return jsonify(response), 200
        response = {
            "Predictions": predictions.tolist()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/train', methods=['POST'])
def train():
    try:
        csv_file_path = request.files['file']
        target_column = request.form.get('target_column', None)

        if target_column is None:
            clusters, score = generic_clustering(csv_file_path)
            model_name = "KMeans"

            response = {
                "Model ": model_name,
                "Model_Type": "Clustering",
                "Optimal No of Clusters ": int(clusters),
                "Score ": float(score),
                "download_link": url_for('download_python_file', model_name=model_name, _external=True)
            }

            return jsonify(response), 200

        else:
            print("CSV file path:", csv_file_path)  # Debugging output
            model_name, best_metric, accuracy, precision, recall, classes = supervised_preprocessing(csv_file_path,
                                                                                                     target_column)

            if accuracy:
                response = {
                    "Model ": model_name,
                    "Classes": classes,
                    "Model Type": "Classification",
                    "F1_Score": best_metric,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "download_link": url_for('download_python_file', model_name=model_name, _external=True)
                }
            else:
                response = {
                    "Model ": model_name,
                    "Classes": classes,
                    "Model Type": "Regression",
                    "R2_Score": best_metric,
                    "download_link": url_for('download_python_file', model_name=model_name, _external=True)
                }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
        # print("Error", e)


if __name__ == '__main__':
    app.run(debug=True)
