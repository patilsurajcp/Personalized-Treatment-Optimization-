import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: NLTK data download failed. Some NLP features may not work properly.")

class NeuralHealthPredictor:
    """A neural network-based health diagnosis predictor with NLP capabilities."""

    def __init__(self, model_path='MLModels/health_nn_model.h5',
                 vectorizer_path='MLModels/vectorizer.pkl',
                 label_encoder_path='MLModels/label_encoder.pkl'):
        """Initialize the neural health predictor."""
        print("Initializing Neural Health Predictor...")

        # Load the dataset
        self.dataset = pd.read_csv('MLModels/Datasets/dataset.csv')
        self.descriptions = pd.read_csv('MLModels/Datasets/symptom_Description.csv')
        self.precautions = pd.read_csv('MLModels/Datasets/symptom_precaution.csv')
        try:
            self.severity = pd.read_csv('MLModels/Datasets/symptom_severity.csv')
        except:
            # Create a default severity dictionary if file not found
            self.severity = pd.DataFrame({'Symptom': self.extract_all_symptoms(), 'weight': [3] * len(self.extract_all_symptoms())})

        # Extract all symptoms
        self.all_symptoms = self.extract_all_symptoms()

        # Create disease-symptom dictionary
        self.disease_symptom_dict = self.create_disease_symptom_dict()

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Model paths
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_encoder_path = label_encoder_path

        # For interactive questioning
        self.confirmed_symptoms = set()
        self.denied_symptoms = set()
        self.asked_symptoms = set()  # Track all asked symptoms to avoid repetition
        self.physical_params = {}
        self.num_questions = 7  # Number of follow-up questions to ask (5-7 questions)

        # Load the model
        self.load_model()

        print("Neural Health Predictor initialized successfully!")

    def extract_all_symptoms(self):
        """Extract all unique symptoms from the dataset."""
        all_symptoms = set()

        # Get all symptom columns
        symptom_cols = [col for col in self.dataset.columns if 'Symptom' in col]

        # Extract unique symptoms
        for col in symptom_cols:
            symptoms = self.dataset[col].dropna().unique()
            all_symptoms.update(symptoms)

        return sorted(list(all_symptoms))

    def create_disease_symptom_dict(self):
        """Create a dictionary mapping diseases to their symptoms."""
        disease_symptom_dict = {}

        # Get all symptom columns
        symptom_cols = [col for col in self.dataset.columns if 'Symptom' in col]

        # Create the dictionary
        for _, row in self.dataset.iterrows():
            disease = row['Disease']
            symptoms = [row[col] for col in symptom_cols if pd.notna(row[col])]

            if disease in disease_symptom_dict:
                disease_symptom_dict[disease].update(symptoms)
            else:
                disease_symptom_dict[disease] = set(symptoms)

        # Convert sets to lists
        for disease in disease_symptom_dict:
            disease_symptom_dict[disease] = sorted(list(disease_symptom_dict[disease]))

        return disease_symptom_dict

    def preprocess_text(self, text):
        """Preprocess text for the model."""
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        words = text.split()

        # Remove stop words and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        # Join back into a string
        return ' '.join(words)

    def load_model(self):
        """Load the pre-trained model and components."""
        try:
            self.model = load_model(self.model_path)

            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            with open(self.label_encoder_path, 'rb') as f:
                encoders = pickle.load(f)
                self.label_to_idx = encoders['label_to_idx']
                self.idx_to_label = encoders['idx_to_label']
                self.diseases = sorted(list(self.label_to_idx.keys()))

            print("Model and components loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_from_text(self, text, top_n=3):
        """Predict diseases from text description using the neural network with improved accuracy."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)

        # First, extract symptoms from the text
        extracted_symptoms = self.extract_symptoms_from_text(text)

        # If we found specific body part symptoms, prioritize diseases related to those symptoms
        body_parts = ['knee', 'ankle', 'elbow', 'shoulder', 'hip', 'neck', 'back', 'head',
                     'chest', 'stomach', 'throat', 'eye', 'ear', 'nose', 'foot', 'hand']

        # Check if any body parts are mentioned in the text
        mentioned_body_parts = [part for part in body_parts if f" {part} " in f" {text.lower()} "]

        # If specific body parts are mentioned, adjust predictions
        body_part_diseases = {
            'knee': ['Osteoarthritis', 'Arthritis'],
            'ankle': ['Osteoarthritis', 'Arthritis'],
            'elbow': ['Osteoarthritis', 'Arthritis'],
            'shoulder': ['Osteoarthritis', 'Arthritis', 'Cervical spondylosis'],
            'hip': ['Osteoarthritis', 'Arthritis'],
            'neck': ['Cervical spondylosis', 'Osteoarthritis'],
            'back': ['Back Pain', 'Osteoarthritis', 'Cervical spondylosis'],
            'head': ['Migraine', 'Tension headache', 'Cluster headache'],
            'chest': ['Bronchial Asthma', 'Pneumonia', 'Heart attack'],
            'stomach': ['Gastroenteritis', 'GERD', 'Peptic ulcer disease'],
            'throat': ['Throat infection', 'Common Cold', 'Streptococcal pharyngitis']
        }

        # Vectorize
        X = self.vectorizer.transform([processed_text]).toarray()

        # Predict
        y_pred_proba = self.model.predict(X)[0]

        # Adjust probabilities based on extracted symptoms and body parts
        if extracted_symptoms and mentioned_body_parts:
            # Get all diseases that could be related to the mentioned body parts
            relevant_diseases = set()
            for part in mentioned_body_parts:
                if part in body_part_diseases:
                    relevant_diseases.update(body_part_diseases[part])

            # Boost probabilities for relevant diseases
            for i, disease in enumerate(self.idx_to_label.values()):
                if disease in relevant_diseases:
                    # Boost the probability significantly
                    y_pred_proba[i] *= 2.0

        # Get top N predictions
        top_indices = np.argsort(y_pred_proba)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            disease = self.idx_to_label[idx]
            confidence = y_pred_proba[idx]

            # Get disease description
            description = ""
            desc_row = self.descriptions[self.descriptions['Disease'] == disease]
            if not desc_row.empty:
                description = desc_row.iloc[0]['Description']

            # Get precautions
            precautions = []
            prec_row = self.precautions[self.precautions['Disease'] == disease]
            if not prec_row.empty:
                for i in range(1, 5):
                    col = f'Precaution_{i}'
                    if col in prec_row.columns and pd.notna(prec_row.iloc[0][col]):
                        precautions.append(prec_row.iloc[0][col])

            # Get common symptoms for this disease
            common_symptoms = self.disease_symptom_dict.get(disease, [])[:5]

            # Get matched symptoms
            matched_symptoms = self.extract_symptoms_from_text(text, disease)

            results.append({
                'disease': disease,
                'confidence': float(confidence),
                'description': description,
                'precautions': precautions,
                'common_symptoms': common_symptoms,
                'matched_symptoms': matched_symptoms
            })

        return results

    def extract_symptoms_from_text(self, text, disease=None):
        """Extract symptoms from text with improved matching."""
        text = " " + text.lower() + " "  # Add spaces to help with word boundary detection
        extracted_symptoms = []

        # If disease is provided, only check symptoms for that disease
        symptoms_to_check = self.disease_symptom_dict.get(disease, []) if disease else self.all_symptoms

        # First pass: Look for exact matches with specific body parts
        specific_matches = []
        body_parts = ['knee', 'ankle', 'elbow', 'shoulder', 'hip', 'neck', 'back', 'head',
                     'chest', 'stomach', 'throat', 'eye', 'ear', 'nose', 'foot', 'hand']

        # Check for body part mentions first (prioritize these)
        mentioned_body_parts = [part for part in body_parts if f" {part} " in text]

        # If specific body parts are mentioned, prioritize symptoms related to those parts
        if mentioned_body_parts:
            for symptom in symptoms_to_check:
                clean_symptom = symptom.replace('_', ' ').lower()
                # Check if the symptom contains any of the mentioned body parts
                if any(part in clean_symptom for part in mentioned_body_parts):
                    if f" {clean_symptom} " in text or any(f" {part} pain" in text for part in mentioned_body_parts if part in clean_symptom):
                        specific_matches.append(symptom)

            # If we found specific matches, return those
            if specific_matches:
                return specific_matches

        # Second pass: Standard symptom matching
        for symptom in symptoms_to_check:
            clean_symptom = symptom.replace('_', ' ').lower()
            # Use word boundary check for more accurate matching
            if f" {clean_symptom} " in text:
                extracted_symptoms.append(symptom)

        # If we found standard matches, return those
        if extracted_symptoms:
            return extracted_symptoms

        # Third pass: Try partial matches with key symptom words
        symptom_keywords = ['pain', 'ache', 'sore', 'hurt', 'fever', 'cough', 'fatigue',
                          'nausea', 'vomit', 'diarrhea', 'rash', 'swelling', 'dizzy']

        # Find mentioned keywords
        for keyword in symptom_keywords:
            if f" {keyword} " in text or f" {keyword}s " in text or f" {keyword}ing " in text:
                # Find symptoms containing this keyword
                for symptom in symptoms_to_check:
                    clean_symptom = symptom.replace('_', ' ').lower()
                    if keyword in clean_symptom and symptom not in extracted_symptoms:
                        extracted_symptoms.append(symptom)

        return extracted_symptoms

    def format_prediction(self, text, predictions):
        """Format the prediction results for display."""
        result = {}

        # Extract symptoms
        extracted_symptoms = []
        for pred in predictions:
            extracted_symptoms.extend(pred['matched_symptoms'])
        extracted_symptoms = list(set(extracted_symptoms))

        if extracted_symptoms:
            result['identified_symptoms'] = [s.replace('_', ' ') for s in extracted_symptoms]
        else:
            result['identified_symptoms'] = []

        # Top prediction
        top_prediction = predictions[0]
        disease = top_prediction['disease']

        result['top_prediction'] = disease
        result['description'] = top_prediction['description']

        # Common symptoms
        result['common_symptoms'] = [s.replace('_', ' ') for s in top_prediction['common_symptoms']]

        # Precautions
        result['precautions'] = top_prediction['precautions']

        # Other possible conditions
        if len(predictions) > 1:
            result['other_conditions'] = [pred['disease'] for pred in predictions[1:]]
        else:
            result['other_conditions'] = []

        return result

    def get_symptom_severity(self, symptom):
        """Get the severity weight of a symptom."""
        row = self.severity[self.severity['Symptom'] == symptom]
        if not row.empty:
            return row.iloc[0]['weight']
        return 3  # Default medium severity

    def get_next_question(self):
        """Get the next best question to ask based on current symptoms with improved relevance."""
        # Make a prediction based on current symptoms
        predicted_disease = None
        if self.confirmed_symptoms:
            # Calculate scores for each disease
            disease_scores = {}
            for disease, disease_symptoms in self.disease_symptom_dict.items():
                matching = self.confirmed_symptoms & set(disease_symptoms)
                denied = self.denied_symptoms & set(disease_symptoms)

                if matching:
                    # Calculate severity-weighted score
                    matching_severity = sum(self.get_symptom_severity(s) for s in matching)
                    total_severity = sum(self.get_symptom_severity(s) for s in disease_symptoms)

                    # Base score is the proportion of severity-weighted symptoms
                    score = matching_severity / total_severity if total_severity > 0 else 0

                    # Penalize for denied symptoms
                    if denied:
                        denied_severity = sum(self.get_symptom_severity(s) for s in denied)
                        penalty = denied_severity / total_severity if total_severity > 0 else 0
                        score = max(0, score - penalty)

                    disease_scores[disease] = score

            # Find the disease with the highest score
            if disease_scores:
                predicted_disease = max(disease_scores.items(), key=lambda x: x[1])[0]

        # Helper function to check if a symptom is valid to ask about
        def is_valid_symptom(symptom):
            return (symptom not in self.confirmed_symptoms and
                    symptom not in self.denied_symptoms and
                    symptom not in self.asked_symptoms)

        # Check for specific body parts in confirmed symptoms
        body_parts = ['knee', 'ankle', 'elbow', 'shoulder', 'hip', 'neck', 'back', 'head',
                     'chest', 'stomach', 'throat', 'eye', 'ear', 'nose', 'foot', 'hand']

        # Track symptom categories to ensure diverse questions
        symptom_categories = {
            'pain': ['pain', 'ache', 'hurt', 'sore', 'tender', 'discomfort'],
            'movement': ['stiffness', 'mobility', 'flexibility', 'range', 'motion', 'walking', 'movement'],
            'appearance': ['swelling', 'redness', 'inflammation', 'bruising', 'discoloration'],
            'sensation': ['numbness', 'tingling', 'burning', 'sensitivity', 'cold', 'hot'],
            'general': ['fatigue', 'weakness', 'tiredness', 'fever', 'chills', 'sweating'],
            'digestive': ['nausea', 'vomiting', 'diarrhea', 'constipation', 'appetite'],
            'respiratory': ['cough', 'breathing', 'shortness', 'wheezing', 'congestion'],
            'neurological': ['dizziness', 'headache', 'vision', 'confusion', 'memory']
        }

        # Track which categories we've already asked about
        asked_categories = set()
        for symptom in self.asked_symptoms:
            symptom_lower = symptom.replace('_', ' ').lower()
            for category, keywords in symptom_categories.items():
                if any(keyword in symptom_lower for keyword in keywords):
                    asked_categories.add(category)

        mentioned_body_parts = []
        for symptom in self.confirmed_symptoms:
            clean_symptom = symptom.replace('_', ' ').lower()
            for part in body_parts:
                if part in clean_symptom:
                    mentioned_body_parts.append(part)

        # If specific body parts are mentioned, prioritize related symptoms
        if mentioned_body_parts:
            # Define related symptoms for each body part - expanded with more symptoms
            body_part_symptoms = {
                'knee': ['knee_pain', 'swelling_joints', 'movement_stiffness', 'painful_walking', 'joint_pain',
                         'muscle_weakness', 'cramps', 'joint_redness', 'joint_swelling', 'difficulty_in_bending'],
                'ankle': ['joint_pain', 'swelling_joints', 'movement_stiffness', 'painful_walking', 'muscle_weakness',
                          'cramps', 'joint_redness', 'joint_swelling', 'difficulty_in_bending'],
                'elbow': ['joint_pain', 'swelling_joints', 'movement_stiffness', 'muscle_weakness', 'joint_redness',
                          'joint_swelling', 'difficulty_in_bending', 'numbness_in_limbs', 'weakness_in_limbs'],
                'shoulder': ['joint_pain', 'neck_pain', 'stiff_neck', 'pain_behind_the_eyes', 'muscle_weakness',
                             'numbness_in_limbs', 'weakness_in_limbs', 'back_pain', 'limited_shoulder_movement'],
                'hip': ['joint_pain', 'swelling_joints', 'movement_stiffness', 'painful_walking', 'muscle_weakness',
                        'difficulty_in_bending', 'back_pain', 'weakness_in_limbs', 'numbness_in_limbs'],
                'neck': ['stiff_neck', 'neck_pain', 'headache', 'dizziness', 'pain_behind_the_eyes', 'muscle_weakness',
                         'numbness_in_limbs', 'weakness_in_limbs', 'back_pain', 'spinning_movements'],
                'back': ['back_pain', 'stiff_neck', 'weakness_in_limbs', 'movement_stiffness', 'muscle_weakness',
                         'numbness_in_limbs', 'painful_walking', 'difficulty_in_bending', 'cramps'],
                'head': ['headache', 'dizziness', 'pain_behind_the_eyes', 'blurred_vision', 'sinus_pressure',
                         'spinning_movements', 'nausea', 'vomiting', 'loss_of_balance', 'lack_of_concentration'],
                'chest': ['chest_pain', 'breathlessness', 'fast_heart_rate', 'cough', 'mucoid_sputum',
                          'phlegm', 'congestion', 'palpitations', 'irregular_heartbeat', 'wheezing'],
                'stomach': ['stomach_pain', 'vomiting', 'nausea', 'abdominal_pain', 'diarrhoea',
                            'constipation', 'loss_of_appetite', 'indigestion', 'acidity', 'stomach_bleeding'],
                'throat': ['throat_irritation', 'cough', 'phlegm', 'swelled_lymph_nodes', 'malaise',
                           'congestion', 'runny_nose', 'sinus_pressure', 'loss_of_smell', 'patches_in_throat']
            }

            # First, try to find symptoms related to the body part from categories we haven't asked about yet
            for part in mentioned_body_parts:
                if part in body_part_symptoms:
                    # Group symptoms by category
                    categorized_symptoms = {}
                    for symptom in body_part_symptoms[part]:
                        if symptom in self.all_symptoms and is_valid_symptom(symptom):
                            symptom_lower = symptom.replace('_', ' ').lower()
                            for category, keywords in symptom_categories.items():
                                if any(keyword in symptom_lower for keyword in keywords):
                                    if category not in asked_categories:
                                        if category not in categorized_symptoms:
                                            categorized_symptoms[category] = []
                                        categorized_symptoms[category].append(symptom)

                    # If we have symptoms from categories we haven't asked about yet, return one
                    if categorized_symptoms:
                        # Choose a category at random
                        import random
                        category = random.choice(list(categorized_symptoms.keys()))
                        return random.choice(categorized_symptoms[category])

            # If we've exhausted categories, just return any valid symptom related to the body part
            for part in mentioned_body_parts:
                if part in body_part_symptoms:
                    valid_symptoms = [s for s in body_part_symptoms[part] if s in self.all_symptoms and is_valid_symptom(s)]
                    if valid_symptoms:
                        return valid_symptoms[0]

        # If we have a predicted disease, use that to guide our questions
        if predicted_disease:
            # Get symptoms for the predicted disease that haven't been asked about
            disease_symptoms = self.disease_symptom_dict.get(predicted_disease, [])

            # Group symptoms by category
            categorized_symptoms = {}
            for symptom in disease_symptoms:
                if is_valid_symptom(symptom):
                    symptom_lower = symptom.replace('_', ' ').lower()
                    for category, keywords in symptom_categories.items():
                        if any(keyword in symptom_lower for keyword in keywords):
                            if category not in asked_categories:
                                if category not in categorized_symptoms:
                                    categorized_symptoms[category] = []
                                categorized_symptoms[category].append(symptom)

            # If we have symptoms from categories we haven't asked about yet, return one
            if categorized_symptoms:
                import random
                category = random.choice(list(categorized_symptoms.keys()))
                return random.choice(categorized_symptoms[category])

            # If we've exhausted categories, just return any valid symptom for the disease
            valid_symptoms = [s for s in disease_symptoms if is_valid_symptom(s)]
            if valid_symptoms:
                return valid_symptoms[0]

        # If we don't have a predicted disease or have exhausted its symptoms,
        # ask about common symptoms but more targeted and diverse
        # Group common symptoms by category
        common_symptoms_by_category = {
            'pain': ['headache', 'joint_pain', 'stomach_pain', 'chest_pain', 'back_pain', 'neck_pain', 'knee_pain'],
            'movement': ['movement_stiffness', 'painful_walking', 'difficulty_in_bending', 'limited_shoulder_movement'],
            'appearance': ['swelling_joints', 'joint_redness', 'joint_swelling', 'skin_rash', 'yellowish_skin'],
            'sensation': ['numbness_in_limbs', 'burning_micturition', 'continuous_sneezing', 'shivering'],
            'general': ['fatigue', 'weakness_in_limbs', 'muscle_weakness', 'high_fever', 'mild_fever'],
            'digestive': ['nausea', 'vomiting', 'diarrhoea', 'constipation', 'loss_of_appetite'],
            'respiratory': ['cough', 'breathlessness', 'phlegm', 'congestion', 'runny_nose'],
            'neurological': ['dizziness', 'loss_of_balance', 'lack_of_concentration', 'blurred_vision']
        }

        # Find categories we haven't asked about yet
        unasked_categories = [cat for cat in common_symptoms_by_category.keys() if cat not in asked_categories]

        # If we have unasked categories, choose one and ask about a symptom from it
        if unasked_categories:
            import random
            category = random.choice(unasked_categories)
            symptoms = common_symptoms_by_category[category]
            valid_symptoms = [s for s in symptoms if s in self.all_symptoms and is_valid_symptom(s)]
            if valid_symptoms:
                return random.choice(valid_symptoms)

        # If we've exhausted all categories or can't find valid symptoms in them,
        # find the most discriminative symptom across all diseases
        candidate_diseases = []
        for disease, symptoms in self.disease_symptom_dict.items():
            if self.confirmed_symptoms & set(symptoms):
                candidate_diseases.append(disease)

        if candidate_diseases:
            # Count how many candidate diseases each symptom appears in
            symptom_counts = {}
            for symptom in self.all_symptoms:
                if not is_valid_symptom(symptom):
                    continue

                count = sum(1 for disease in candidate_diseases if symptom in self.disease_symptom_dict.get(disease, []))
                if count > 0:
                    # The closer to half the diseases, the more discriminative
                    symptom_counts[symptom] = abs(count - len(candidate_diseases) / 2)

            # Get the most discriminative symptom
            sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1])
            if sorted_symptoms:
                return sorted_symptoms[0][0]

        # Last resort: return any valid symptom
        valid_symptoms = [s for s in self.all_symptoms if is_valid_symptom(s)]
        if valid_symptoms:
            import random
            return random.choice(valid_symptoms[:20])  # Choose from first 20 to be more predictable

        # If we've asked all possible questions or can't find a good next question
        return None

    def format_question(self, symptom):
        """Format a doctor-like question about a symptom."""
        clean_symptom = symptom.replace('_', ' ').lower()

        # Doctor-like question templates
        templates = [
            f"Are you experiencing {clean_symptom}?",
            f"Have you noticed any {clean_symptom}?",
            f"Do you have {clean_symptom}?"
        ]

        import random
        return random.choice(templates)

    def diagnose(self, text, interactive=False):
        """Process user input and return a diagnosis using the neural network."""
        # Reset symptoms for a new diagnosis
        self.confirmed_symptoms = set()
        self.denied_symptoms = set()
        self.asked_symptoms = set()  # Reset asked symptoms

        # Extract initial symptoms from text
        initial_symptoms = self.extract_symptoms_from_text(text)
        self.confirmed_symptoms.update(initial_symptoms)

        # If not interactive, just return the prediction based on the text
        if not interactive:
            predictions = self.predict_from_text(text)
            return self.format_prediction(text, predictions)

        # For interactive mode, we'll return a dict with the initial symptoms and the next question
        next_symptom = self.get_next_question()
        if next_symptom:
            next_question = self.format_question(next_symptom)
            # Mark this symptom as about to be asked
            self.asked_symptoms.add(next_symptom)
        else:
            next_question = None

        # Make an initial prediction
        predictions = self.predict_from_text(text) if text.strip() else self.predict_from_text("general checkup")
        formatted_result = self.format_prediction(text, predictions)

        # Add the interactive elements
        formatted_result['initial_symptoms'] = [s.replace('_', ' ') for s in initial_symptoms]
        formatted_result['next_question'] = next_question
        formatted_result['next_symptom'] = next_symptom
        formatted_result['question_count'] = 0
        formatted_result['asked_symptoms'] = [s.replace('_', ' ') for s in self.asked_symptoms]

        return formatted_result

    def answer_question(self, symptom, answer, current_diagnosis):
        """Process the answer to a question and update the diagnosis."""
        try:
            # Mark this symptom as asked
            self.asked_symptoms.add(symptom)

            # Update confirmed/denied symptoms based on the answer
            if answer.lower() in ['yes', 'y', 'true', '1']:
                self.confirmed_symptoms.add(symptom)
            elif answer.lower() in ['no', 'n', 'false', '0']:
                self.denied_symptoms.add(symptom)

            # Get the next question
            next_symptom = self.get_next_question()
            if next_symptom:
                next_question = self.format_question(next_symptom)
            else:
                next_question = None

            # Make a new prediction
            # Create a text representation of the symptoms for the model
            symptom_text = ' '.join([s.replace('_', ' ') for s in self.confirmed_symptoms]) if self.confirmed_symptoms else "general checkup"
            predictions = self.predict_from_text(symptom_text)
            formatted_result = self.format_prediction(symptom_text, predictions)

            # Update the question count
            question_count = current_diagnosis.get('question_count', 0) + 1

            # Add the interactive elements
            formatted_result['initial_symptoms'] = current_diagnosis.get('initial_symptoms', [])
            formatted_result['next_question'] = next_question
            formatted_result['next_symptom'] = next_symptom
            formatted_result['question_count'] = question_count
            formatted_result['confirmed_symptoms'] = [s.replace('_', ' ') for s in self.confirmed_symptoms]
            formatted_result['denied_symptoms'] = [s.replace('_', ' ') for s in self.denied_symptoms]
            formatted_result['asked_symptoms'] = [s.replace('_', ' ') for s in self.asked_symptoms]

            # If we've asked exactly 3 questions or there are no more questions, mark as complete
            if question_count >= 3 or not next_question:
                formatted_result['diagnosis_complete'] = True
            else:
                formatted_result['diagnosis_complete'] = False

            return formatted_result
        except Exception as e:
            # If there's an error, return a diagnosis with an error message
            print(f"Error in answer_question: {str(e)}")

            # Create a basic result with error information
            error_result = {
                'error': str(e),
                'diagnosis_complete': True,
                'top_prediction': 'Unable to determine',
                'description': f'An error occurred while processing your answer: {str(e)}',
                'common_symptoms': [],
                'precautions': ['Please try again with different symptoms'],
                'other_conditions': [],
                'identified_symptoms': [s.replace('_', ' ') for s in self.confirmed_symptoms] if hasattr(self, 'confirmed_symptoms') else [],
                'question_count': current_diagnosis.get('question_count', 0) + 1
            }

            return error_result

# Singleton instance
_predictor = None

def get_predictor():
    """Get or create the predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = NeuralHealthPredictor()
    return _predictor
