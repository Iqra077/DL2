import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Input, Embedding, Flatten, Dot, Dense, Concatenate, LSTM, Bidirectional, 
    Attention, Dropout, BatchNormalization
)
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify

# Load datasets
student_performance_path = 'student_performance_data.csv'
study_resources_path = 'study_resources_data.csv'
student_data = pd.read_csv(student_performance_path)
resource_data = pd.read_csv(study_resources_path)

# Preprocessing
def preprocess_data(student_data, resource_data):
    le_student = LabelEncoder()
    le_quiz = LabelEncoder()
    le_topic = LabelEncoder()
    le_difficulty = LabelEncoder()

    student_data['Student_ID'] = le_student.fit_transform(student_data['Student_ID'])
    student_data['Quiz_ID'] = le_quiz.fit_transform(student_data['Quiz_ID'])
    student_data['Topic'] = le_topic.fit_transform(student_data['Topic'])
    student_data['Difficulty'] = le_difficulty.fit_transform(student_data['Difficulty'])

    scaler = MinMaxScaler()
    student_data[['Score', 'Time_Taken']] = scaler.fit_transform(student_data[['Score', 'Time_Taken']])
    return student_data, resource_data, le_student, le_quiz, le_topic

student_data, resource_data, le_student, le_quiz, le_topic = preprocess_data(student_data, resource_data)

# Clustering for performance levels
def cluster_students(student_data):
    student_avg_performance = student_data.groupby('Student_ID')[['Score']].mean().reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42)
    student_avg_performance['Cluster'] = kmeans.fit_predict(student_avg_performance[['Score']])
    performance_levels = {0: 'Weak', 1: 'Average', 2: 'Good'}
    student_avg_performance['Performance_Level'] = student_avg_performance['Cluster'].map(performance_levels)
    return student_avg_performance

student_avg_performance = cluster_students(student_data)

# Multi-task Learning Model
def build_multi_task_model(num_students, num_quizzes):
    # Inputs
    student_input = Input(shape=(1,), name='Student_Input')
    quiz_input = Input(shape=(1,), name='Quiz_Input')

    # Embeddings
    student_embedding = Embedding(input_dim=num_students, output_dim=50, name='Student_Embedding')(student_input)
    quiz_embedding = Embedding(input_dim=num_quizzes, output_dim=50, name='Quiz_Embedding')(quiz_input)

    # Flatten
    student_vec = Flatten(name='Flatten_Students')(student_embedding)
    quiz_vec = Flatten(name='Flatten_Quizzes')(quiz_embedding)

    # Attention Mechanism
    attention = Attention(name="Attention_Layer")([student_vec, quiz_vec])
    
    # Combine embeddings and attention
    combined = Concatenate()([student_vec, quiz_vec, attention])
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Multi-task outputs
    score_output = Dense(1, activation='sigmoid', name='Score_Output')(x)
    time_output = Dense(1, activation='relu', name='Time_Output')(x)

    # Model
    model = Model(inputs=[student_input, quiz_input], outputs=[score_output, time_output])
    model.compile(optimizer='adam', loss={'Score_Output': 'binary_crossentropy', 'Time_Output': 'mse'}, 
                  metrics={'Score_Output': 'accuracy', 'Time_Output': 'mae'})
    return model

# Train-Test Split
train, test = train_test_split(student_data[['Student_ID', 'Quiz_ID', 'Score', 'Time_Taken']], test_size=0.2, random_state=42)
num_students = student_data['Student_ID'].nunique()
num_quizzes = student_data['Quiz_ID'].nunique()

model = build_multi_task_model(num_students, num_quizzes)
model.fit(
    [train['Student_ID'], train['Quiz_ID']], 
    {'Score_Output': train['Score'], 'Time_Output': train['Time_Taken']}, 
    validation_data=(
        [test['Student_ID'], test['Quiz_ID']], 
        {'Score_Output': test['Score'], 'Time_Output': test['Time_Taken']}
    ), 
    epochs=10, batch_size=64
)

# Sequential Embeddings with LSTMs for Topics
def create_topic_embeddings(student_data):
    topic_sequences = student_data.groupby('Student_ID')['Topic'].apply(list)
    max_len = max(topic_sequences.apply(len))
    topic_input = Input(shape=(max_len,), name='Topic_Input')
    topic_embedding = Embedding(input_dim=student_data['Topic'].nunique(), output_dim=50, name='Topic_Embedding')(topic_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(topic_embedding)
    return Model(inputs=topic_input, outputs=lstm_out)

topic_model = create_topic_embeddings(student_data)

# Recommendations Generation with Dynamic Resource Adjustment
def generate_recommendations(student_avg_performance, resource_data, le_topic, performance_levels):
    recommendations = {}
    for student_id, performance_level in zip(student_avg_performance['Student_ID'], student_avg_performance['Performance_Level']):
        student_topics = student_data[student_data['Student_ID'] == student_id].groupby('Topic')[['Score']].mean().reset_index()
        weak_topics = student_topics[student_topics['Score'] < 0.4]['Topic']
        
        if performance_level == 'Weak':
            resources = resource_data[resource_data['Topic'].isin(le_topic.inverse_transform(weak_topics))].groupby('Topic').apply(
                lambda x: x.sample(n=min(len(x), 7), random_state=42)).reset_index(drop=True)
        elif performance_level == 'Average':
            resources = resource_data.groupby('Topic').apply(lambda x: x.sample(n=min(len(x), 6), random_state=42)).reset_index(drop=True)
        else:
            resources = resource_data.groupby('Topic').apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)).reset_index(drop=True)
        
        recommendations[le_student.inverse_transform([student_id])[0]] = resources[['Resource_ID', 'Type', 'Platform', 'Resource_Link']].to_dict('records')
    return recommendations

recommendations = generate_recommendations(student_avg_performance, resource_data, le_topic, performance_levels)



# Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    """
    Render the main index page.
    """
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Fetch recommendations for a specific student ID.
    """
    student_id = request.args.get('student_id')
    
    # Validate student_id
    if not student_id:
        logging.warning("Student ID not provided.")
        return jsonify({'error': 'Student ID is required'}), 400

    try:
        transformed_id = le_student.transform([student_id])[0]  # Transform Student ID
    except ValueError:
        logging.error(f"Student ID {student_id} is invalid.")
        return jsonify({'error': f"Student ID '{student_id}' not found."}), 404

    if transformed_id in recommendations:
        performance_level = student_avg_performance[
            student_avg_performance['Student_ID'] == transformed_id
        ]['Performance_Level'].values[0]
        
        logging.info(f"Recommendations fetched for Student ID: {student_id}")
        return jsonify({
            'student_id': student_id,
            'performance_level': performance_level,
            'recommendations': recommendations[student_id]
        })
    else:
        logging.error(f"No recommendations available for Student ID: {student_id}.")
        return jsonify({'error': f"No recommendations found for Student ID: {student_id}"}), 404

@app.route('/students', methods=['GET'])
def get_students():
    """
    Fetch all valid Student IDs for auto-completion or selection in the frontend.
    """
    student_ids = le_student.inverse_transform(student_avg_performance['Student_ID'])
    return jsonify({'students': list(student_ids)})

@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 errors for non-existent routes.
    """
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle internal server errors.
    """
    logging.exception("An internal error occurred.")
    return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8002)
