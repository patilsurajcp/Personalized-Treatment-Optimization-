from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import json
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import sys
import os

# Import the health predictor
from health_predictor import get_predictor

# Create your views here.
def Home(request):
    return render(request, 'homepages/home.html')

@login_required
def Ask(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request body
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # Process the message using our health model
            result = process_message(user_message)

            return JsonResponse(result)
        except Exception as e:
            print(f"Error processing message: {str(e)}", file=sys.stderr)
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return render(request, 'homepages/simple_buttons.html')


def answer_question(request):
    """Handle form-based answers to questions"""
    if request.method == 'POST':
        try:
            symptom = request.POST.get('symptom', '')
            answer = request.POST.get('answer', '')
            question_count = int(request.POST.get('question_count', '0'))

            print(f"Received answer: {answer} for symptom: {symptom}, current question count: {question_count}")

            # Get the predictor
            predictor = get_predictor()

            # Create a diagnosis object with the current question count
            current_diagnosis = {
                'question_count': question_count,  # Use the count from the form
                'initial_symptoms': [],
                'confirmed_symptoms': [],
                'denied_symptoms': []
            }

            # Process the answer
            diagnosis = predictor.answer_question(symptom, answer, current_diagnosis)

            # Get the updated question count - should be incremented by 1
            new_question_count = diagnosis.get('question_count', 0)
            print(f"Updated question count: {new_question_count}")

            # Format the response as HTML
            if new_question_count < 7 and diagnosis.get('next_question'):
                # Format the next question
                next_symptom = diagnosis.get('next_symptom', '')
                next_question = diagnosis.get('next_question', '')

                html = f"<div class='diagnosis-result'>"
                html += f"<p><strong>Based on your answers so far, you may have:</strong> {diagnosis['top_prediction']}</p>"
                html += f"<p><small>Question {new_question_count} of 3</small></p>"
                html += f"<div class='question-container' data-symptom='{next_symptom}'>"
                html += f"<p class='question'><strong>{next_question}</strong></p>"
                html += f"<div class='answer-buttons'>"
                html += f"<button class='yes-btn' onclick='handleAnswer(\"{next_symptom}\", \"yes\", {new_question_count})'>"
                html += "Yes</button>"
                html += f"<button class='no-btn' onclick='handleAnswer(\"{next_symptom}\", \"no\", {new_question_count})'>"
                html += "No</button>"
                html += "</div></div></div>"
            else:
                # Final diagnosis
                html = format_final_diagnosis_html(diagnosis)

            return HttpResponse(html)
        except Exception as e:
            print(f"Error in answer_question: {str(e)}", file=sys.stderr)
            return HttpResponse(f"<p>Error: {str(e)}</p>")
    else:
        return HttpResponse("<p>Invalid request method</p>")


def process_message(message):
    # Initialize the health predictor if needed
    try:
        predictor = get_predictor()

        # Check if this is an answer to a previous question
        if message.startswith('ANSWER:'):
            try:
                # Format: ANSWER:symptom:response:diagnosis_json
                parts = message.split(':', 3)
                if len(parts) >= 4:
                    symptom = parts[1]
                    answer = parts[2]

                    # Handle potential JSON parsing errors
                    try:
                        current_diagnosis = json.loads(parts[3])
                    except json.JSONDecodeError as json_err:
                        print(f"JSON decode error: {str(json_err)}", file=sys.stderr)
                        print(f"JSON string: {parts[3][:100]}...", file=sys.stderr)
                        # Create a minimal diagnosis object
                        current_diagnosis = {
                            'question_count': 0,
                            'initial_symptoms': [],
                            'confirmed_symptoms': [],
                            'denied_symptoms': []
                        }

                    # Process the answer and get updated diagnosis
                    diagnosis = predictor.answer_question(symptom, answer, current_diagnosis)

                    # Format the response
                    response = {
                        'user_input': answer,
                        'diagnosis': diagnosis,
                        'is_answer': True
                    }

                    # If we have more questions to ask, include the next question
                    if not diagnosis.get('diagnosis_complete', False) and diagnosis.get('next_question'):
                        response['response'] = format_question_html(diagnosis['next_question'], diagnosis['next_symptom'], diagnosis)
                    else:
                        # Final diagnosis after all questions
                        response['response'] = format_final_diagnosis_html(diagnosis)

                    return response
                else:
                    raise ValueError(f"Invalid answer format: {message[:50]}...")
            except Exception as answer_err:
                print(f"Error processing answer: {str(answer_err)}", file=sys.stderr)
                return {
                    'response': f"I'm sorry, I couldn't process your answer. Let's start over with a new question."
                }

        # This is a new query, start interactive diagnosis
        diagnosis = predictor.diagnose(message, interactive=True)

        # Format the initial response
        response = {
            'user_input': message,
            'diagnosis': diagnosis,
            'is_initial': True
        }

        # Include the first question if available
        if diagnosis.get('next_question'):
            response['response'] = format_initial_response_html(message, diagnosis)
        else:
            # No follow-up questions needed
            response['response'] = format_response_html(message, diagnosis)

        return response
    except Exception as e:
        print(f"Error in process_message: {str(e)}", file=sys.stderr)
        error_html = f"<div class='diagnosis-result'>"
        error_html += f"<p>I'm sorry, I encountered an error while processing your request.</p>"
        error_html += f"<p>Please try again with a different description of your symptoms.</p>"
        error_html += f"<p><em>Error details: {str(e)}</em></p>"
        error_html += f"</div>"

        return {
            'response': error_html
        }


def format_initial_response_html(message, diagnosis):
    """Format the initial response with identified symptoms and first question."""
    html = f"<div class='diagnosis-result'>"

    # Initial symptoms identified
    if diagnosis['initial_symptoms']:
        html += f"<p><strong>I've identified these symptoms:</strong> {', '.join(diagnosis['initial_symptoms'])}</p>"
        html += "<p>Let me ask a few more questions to refine my assessment.</p>"
    else:
        html += "<p>I need more specific information about your symptoms. Let me ask a few questions.</p>"

    # First question
    if diagnosis.get('next_question'):
        symptom = diagnosis.get('next_symptom', '')
        question = diagnosis.get('next_question', '')
        question_count = 1  # First question is always 1

        # Create a unique ID for this question container
        question_id = f"question_{symptom.replace(' ', '_')}"

        html += f"<p><small>Question {question_count} of 7</small></p>"
        html += f"<div id='{question_id}' class='question-container' data-symptom='{symptom}'>"
        html += f"<p class='question'><strong>{question}</strong></p>"
        html += f"<div class='answer-buttons'>"
        html += f"<button class='yes-btn' onclick='handleAnswer(\"{symptom}\", \"yes\", {question_count})'>Yes</button>"
        html += f"<button class='no-btn' onclick='handleAnswer(\"{symptom}\", \"no\", {question_count})'>No</button>"
        html += "</div></div>"

    html += "</div>"
    return html

def format_question_html(question, symptom, diagnosis):
    """Format a follow-up question as HTML."""
    html = f"<div class='diagnosis-result'>"

    # Updated prediction based on answers so far
    html += f"<p><strong>Based on your answers so far, you may have:</strong> {diagnosis['top_prediction']}</p>"

    # Next question
    if question:
        # Get the current question count
        question_count = diagnosis.get('question_count', 0) + 1

        # Create a unique ID for this question container
        question_id = f"question_{symptom.replace(' ', '_')}_{question_count}"

        html += f"<p><small>Question {question_count} of 7</small></p>"
        html += f"<div id='{question_id}' class='question-container' data-symptom='{symptom}'>"
        html += f"<p class='question'><strong>{question}</strong></p>"
        html += f"<div class='answer-buttons'>"
        html += f"<button class='yes-btn' onclick='handleAnswer(\"{symptom}\", \"yes\", {question_count})'>Yes</button>"
        html += f"<button class='no-btn' onclick='handleAnswer(\"{symptom}\", \"no\", {question_count})'>No</button>"
        html += "</div></div>"

    html += "</div>"
    return html

def format_final_diagnosis_html(diagnosis):
    """Format the final diagnosis after all questions have been answered."""
    html = f"<div class='diagnosis-result'>"

    # Summary of the conversation
    html += "<p><strong>Based on our conversation, I've gathered the following information:</strong></p>"

    # Confirmed symptoms
    if diagnosis.get('confirmed_symptoms'):
        html += f"<p><strong>Confirmed symptoms:</strong> {', '.join(diagnosis['confirmed_symptoms'])}</p>"

    # Denied symptoms
    if diagnosis.get('denied_symptoms'):
        html += f"<p><strong>Symptoms you don't have:</strong> {', '.join(diagnosis['denied_symptoms'])}</p>"

    # Top prediction with higher confidence
    html += f"<p><strong>My assessment:</strong> You most likely have <span class='diagnosis-highlight'>{diagnosis['top_prediction']}</span></p>"

    # Description
    if diagnosis['description']:
        html += f"<p><strong>Description:</strong> {diagnosis['description']}</p>"

    # Common symptoms
    if diagnosis['common_symptoms']:
        html += f"<p><strong>Other common symptoms:</strong></p><ul>"
        for symptom in diagnosis['common_symptoms']:
            html += f"<li>{symptom}</li>"
        html += "</ul>"

    # Precautions
    if diagnosis['precautions']:
        html += f"<p><strong>Recommended precautions:</strong></p><ol>"
        for precaution in diagnosis['precautions']:
            html += f"<li>{precaution}</li>"
        html += "</ol>"

    # Other conditions
    if diagnosis['other_conditions']:
        html += f"<p><strong>Other possible conditions:</strong></p><ul>"
        for condition in diagnosis['other_conditions']:
            html += f"<li>{condition}</li>"
        html += "</ul>"

    html += "<p><em>Note: This is not a medical diagnosis. Please consult a healthcare professional.</em></p>"
    html += "</div>"

    return html

def format_response_html(message, diagnosis):
    """Format the diagnosis as HTML for display in the chat."""
    html = f"<div class='diagnosis-result'>"

    # Top prediction
    html += f"<p><strong>Based on your symptoms, you may have:</strong> {diagnosis['top_prediction']}</p>"

    # Identified symptoms
    if diagnosis['identified_symptoms']:
        html += f"<p><strong>Identified symptoms:</strong> {', '.join(diagnosis['identified_symptoms'])}</p>"
    else:
        html += f"<p>No specific symptoms were identified from your description.</p>"

    # Description
    if diagnosis['description']:
        html += f"<p><strong>Description:</strong> {diagnosis['description']}</p>"

    # Common symptoms
    if diagnosis['common_symptoms']:
        html += f"<p><strong>Other common symptoms:</strong></p><ul>"
        for symptom in diagnosis['common_symptoms']:
            html += f"<li>{symptom}</li>"
        html += "</ul>"

    # Precautions
    if diagnosis['precautions']:
        html += f"<p><strong>Recommended precautions:</strong></p><ol>"
        for precaution in diagnosis['precautions']:
            html += f"<li>{precaution}</li>"
        html += "</ol>"

    # Other conditions
    if diagnosis['other_conditions']:
        html += f"<p><strong>Other possible conditions:</strong></p><ul>"
        for condition in diagnosis['other_conditions']:
            html += f"<li>{condition}</li>"
        html += "</ul>"

    html += "<p><em>Note: This is not a medical diagnosis. Please consult a healthcare professional.</em></p>"
    html += "</div>"

    return html
