from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'adaboost_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Attractive Frontend using modern Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Developer Analytics Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    </style>
</head>
<body class="text-slate-200 min-h-screen flex items-center justify-center p-6">
    <div class="bg-slate-900/80 backdrop-blur-md border border-slate-800 p-8 rounded-2xl shadow-2xl max-w-2xl w-full">
        <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 mb-2">DevPerformance AI</h1>
        <p class="text-slate-400 mb-6 text-sm">Input your metrics below to evaluate performance or stress insights using your trained AdaBoost model.</p>
        
        <form id="predictionForm" class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Hours Coding</label>
                <input type="number" step="any" name="Hours_Coding" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">AI Usage Hours</label>
                <input type="number" step="any" name="AI_Usage_Hours" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Lines of Code</label>
                <input type="number" step="any" name="Lines_of_Code" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Commits</label>
                <input type="number" step="any" name="Commits" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Bugs Reported</label>
                <input type="number" step="any" name="Bugs_Reported" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Sleep Hours</label>
                <input type="number" step="any" name="Sleep_Hours" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Distractions</label>
                <input type="number" step="any" name="Distractions" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Cognitive Load</label>
                <input type="number" step="any" name="Cognitive_Load" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            <div class="md:col-span-2">
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Stress Level</label>
                <input type="number" step="any" name="Stress_Level" required class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 transition">
            </div>
            
            <button type="submit" class="md:col-span-2 mt-4 bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-600 hover:to-indigo-600 text-white font-bold py-3 px-4 rounded-xl shadow-lg transform active:scale-98 transition cursor-pointer text-center">
                Generate Prediction
            </button>
        </form>

        <div id="resultContainer" class="hidden mt-6 p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p class="text-sm text-slate-400">Prediction Result</p>
            <h2 id="resultValue" class="text-3xl font-black text-cyan-400 mt-1">--</h2>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Map inputs to exact float array order required by model
            const features = [
                parseFloat(data.Hours_Coding),
                parseFloat(data.AI_Usage_Hours),
                parseFloat(data.Lines_of_Code),
                parseFloat(data.Commits),
                parseFloat(data.Bugs_Reported),
                parseFloat(data.Sleep_Hours),
                parseFloat(data.Distractions),
                parseFloat(data.Cognitive_Load),
                parseFloat(data.Stress_Level)
            ];

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features })
                });
                const result = await response.json();
                
                const container = document.getElementById('resultContainer');
                const valueDisplay = document.getElementById('resultValue');
                
                if(result.prediction !== undefined) {
                    valueDisplay.innerText = result.prediction;
                    container.classList.remove('hidden');
                } else {
                    alert("Error: " + result.error);
                }
            } catch (err) {
                alert("Failed to connect to the prediction engine.");
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
