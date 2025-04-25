from flask import Flask, request, jsonify, render_template
import os
import joblib

app = Flask(__name__)

model_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(model_dir, 'iris_model.pkl')

try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    print("File not found!")
    exit()

@app.route('/', methods=['GET', 'POST'])
def predict():
    species=''
    if request.method == 'POST':
        data = [float(request.form['sl']), float(request.form['sw']),
                float(request.form['pl']), float(request.form['pw'])]
        prediction = model.predict([data])[0]
        species = ['Setosa', 'Versicolor', 'Virginica'][prediction]

    return render_template('index.html', species=species)

if __name__ == '__main__':
    app.run(debug=True)