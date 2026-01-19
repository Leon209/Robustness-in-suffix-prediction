from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import csv
import io
import sys
import os
from datetime import datetime

# Add the project root to the path to import robustness modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from robustness.helpdesk_perturbations import HelpdeskPerturbator

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    try:
        data = request.get_json()
        dataset = data.get('dataset')
        perturbable_features = data.get('perturbable_features', [])
        num_perturbations = data.get('num_perturbations')
        perturbation_rate = data.get('perturbation_rate')
        
        # Validate input
        if not dataset:
            return jsonify({'error': 'Dataset is required'}), 400
        
        if dataset != 'Helpdesk':
            return jsonify({'error': 'Only Helpdesk dataset is currently supported'}), 400
        
        if not perturbable_features or len(perturbable_features) == 0:
            return jsonify({'error': 'At least one perturbable feature is required'}), 400
        
        if not num_perturbations or num_perturbations < 1 or num_perturbations > len(perturbable_features):
            return jsonify({'error': f'Number of perturbations must be between 1 and {len(perturbable_features)}'}), 400
        
        if not perturbation_rate or perturbation_rate <= 0 or perturbation_rate > 1:
            return jsonify({'error': 'Perturbation rate must be between 0 and 1 (exclusive of 0)'}), 400
        
        # Get the path to the helpdesk CSV file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, 'data', 'helpdesk.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': f'Dataset file not found: {csv_path}'}), 404
        
        # Initialize the perturbator
        perturbator = HelpdeskPerturbator(csv_path=csv_path)

        
        # Apply perturbations using all_events_attack
        perturbed_df, impact = perturbator.all_events_attack(
            num_features_to_perturb=num_perturbations,
            perturbable_features=perturbable_features,
            perturbation_probability=perturbation_rate,
            perturbation_type='random',
        )
        
        # Convert DataFrame to CSV in memory
        output = io.StringIO()
        perturbed_df.to_csv(output, index=False)
        output.seek(0)
        
        # Convert StringIO to BytesIO for file download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_str = '_'.join([f.replace(' ', '_') for f in perturbable_features[:3]])  # First 3 features for filename
        if len(perturbable_features) > 3:
            features_str += f'_plus{len(perturbable_features)-3}'
        filename = f'helpdesk_perturbed_{features_str}_{num_perturbations}feat_{perturbation_rate}rate_{timestamp}.csv'
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
