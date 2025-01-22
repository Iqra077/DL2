import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging

# Load datasets
student_performance_path = 'student_performance_data.csv'
study_resources_path = 'study_resources_data.csv'
student_data = pd.read_csv(student_performance_path)
resource_data = pd.read_csv(study_resources_path)

# Preprocessing student performance data
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

# Cluster students based on their performance
student_avg_performance = student_data.groupby('Student_ID')[['Score']].mean().reset_index()
kmeans = KMeans(n_clusters=3, random_state=42)
student_avg_performance['Cluster'] = kmeans.fit_predict(student_avg_performance[['Score']])

# Define performance levels
performance_levels = {
    0: 'Weak',
    1: 'Average',
    2: 'Good'
}
student_avg_performance['Performance_Level'] = student_avg_performance['Cluster'].map(performance_levels)

# Prepare data for collaborative filtering
student_quiz_matrix = student_data[['Student_ID', 'Quiz_ID', 'Score']]
train, test = train_test_split(student_quiz_matrix, test_size=0.2, random_state=42)

# Neural Collaborative Filtering Model
num_students = student_data['Student_ID'].nunique()
num_quizzes = student_data['Quiz_ID'].nunique()

# Input layers
student_input = Input(shape=(1,), name='Student_Input')
quiz_input = Input(shape=(1,), name='Quiz_Input')

# Embedding layers
student_embedding = Embedding(input_dim=num_students, output_dim=50, name='Student_Embedding')(student_input)
quiz_embedding = Embedding(input_dim=num_quizzes, output_dim=50, name='Quiz_Embedding')(quiz_input)

# Flatten embeddings
student_vec = Flatten(name='Flatten_Students')(student_embedding)
quiz_vec = Flatten(name='Flatten_Quizzes')(quiz_embedding)

# Dot product to predict relevance
dot_product = Dot(axes=1, name='Dot_Product')([student_vec, quiz_vec])

# Dense layers for additional learning
x = Dense(128, activation='relu')(dot_product)
x = Dense(64, activation='relu')(x)
prediction = Dense(1, activation='sigmoid', name='Prediction')(x)

# Model definition
model = Model(inputs=[student_input, quiz_input], outputs=prediction)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training
model.fit([train['Student_ID'], train['Quiz_ID']], train['Score'], 
          validation_data=([test['Student_ID'], test['Quiz_ID']], test['Score']), 
          epochs=10, batch_size=64)

# Content-based Filtering: Match weak topics to study resources
# Content-based Filtering: Match topics to study resources based on performance
recommendations = {}
for student_id, performance_level in zip(student_avg_performance['Student_ID'], student_avg_performance['Performance_Level']):
    # Get the topics the student has interacted with and their scores
    student_topics = student_data[student_data['Student_ID'] == student_id].groupby('Topic')[['Score']].mean().reset_index()
    weak_topics = student_topics[student_topics['Score'] < 0.4]['Topic']
    
    if performance_level == 'Weak':
        resources = resource_data[resource_data['Topic'].isin(le_topic.inverse_transform(weak_topics))]
        # Recommend 6-7 resources per topic
        resources = resources.groupby('Topic').apply(lambda x: x.sample(n=min(len(x), 7), random_state=42)).reset_index(drop=True)
    elif performance_level == 'Average':
        resources = resource_data[resource_data['Topic'].isin(le_topic.inverse_transform(student_topics['Topic']))]
        # Recommend 5-6 resources per topic
        resources = resources.groupby('Topic').apply(lambda x: x.sample(n=min(len(x), 6), random_state=42)).reset_index(drop=True)
    else:  # Good
        resources = resource_data[resource_data['Topic'].isin(le_topic.inverse_transform(student_topics['Topic']))]
        # Recommend 2-3 resources per topic
        resources = resources.groupby('Topic').apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)).reset_index(drop=True)
    
    # Include Resource_ID and other necessary fields in recommendations
    recommendations[le_student.inverse_transform([student_id])[0]] = resources[['Resource_ID', 'Type', 'Platform', 'Resource_Link']].to_dict('records')

# Flask App for Frontend
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    student_id = request.args.get('student_id')
    if student_id in recommendations:
        return jsonify({
            'student_id': student_id,
            'performance_level': student_avg_performance[student_avg_performance['Student_ID'] == le_student.transform([student_id])[0]]['Performance_Level'].values[0],
            'recommendations': recommendations[student_id]
        })
    else:
        return jsonify({'error': 'Student not found'})

if __name__ == '__main__':
    app.run(debug=True, port=8080) 
