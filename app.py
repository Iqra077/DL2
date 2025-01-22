import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Dropout, LSTM, GRU, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Load datasets
performance_data = pd.read_csv('student_performance_data.csv')
resources_data = pd.read_csv('study_resources_data.csv')

# Preprocessing
subjects = ['English', 'Chemistry', 'History', 'Geography', 'Biology', 'Math', 'Physics']

# Normalize subject performance data
performance_matrix = performance_data.pivot_table(index='Student_ID', columns='Topic', values='Score').fillna(0)
scaler = StandardScaler()
performance_matrix = pd.DataFrame(scaler.fit_transform(performance_matrix), columns=performance_matrix.columns, index=performance_matrix.index)

# Label encoding for student IDs and topics (careers or resources)
student_encoder = LabelEncoder()
topic_encoder = LabelEncoder()

performance_data['Student_ID'] = student_encoder.fit_transform(performance_data['Student_ID'])
performance_data['Topic'] = topic_encoder.fit_transform(performance_data['Topic'])

# NCF Model: Neural Collaborative Filtering
def build_ncf_model(num_students, num_topics, input_dim=10):
    student_input = Input(shape=(1,), name='student_id')
    topic_input = Input(shape=(1,), name='topic_id')
    
    student_embedding = Embedding(num_students, input_dim, name='student_embedding')(student_input)
    topic_embedding = Embedding(num_topics, input_dim, name='topic_embedding')(topic_input)
    
    student_vec = Flatten()(student_embedding)
    topic_vec = Flatten()(topic_embedding)
    
    concatenated = Concatenate()([student_vec, topic_vec])
    
    dense_layer = Dense(128, activation='relu')(concatenated)
    dense_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)
    output = Dense(1, activation='linear')(dense_layer)
    
    model = Model(inputs=[student_input, topic_input], outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    
    return model

# LSTM/GRU Model: Sequence Modeling
def build_sequence_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    
    return model

# Prepare data for the models
X_ncf_student = performance_data['Student_ID'].values
X_ncf_topic = performance_data['Topic'].values
y_ncf = performance_data['Score'].values

# For Sequence Modeling
X_seq = performance_matrix.values.reshape((performance_matrix.shape[0], performance_matrix.shape[1], 1))  # [samples, timesteps, features]
y_seq = performance_matrix.values[:, -1]  # Last performance score as target (or could be time-series prediction)

# Train-test split for NCF
X_ncf_student_train, X_ncf_student_test, X_ncf_topic_train, X_ncf_topic_test, y_ncf_train, y_ncf_test = train_test_split(
    X_ncf_student, X_ncf_topic, y_ncf, test_size=0.2, random_state=42)

# Train-test split for Sequence Model
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build and train NCF model
num_students = len(student_encoder.classes_)
num_topics = len(topic_encoder.classes_)
ncf_model = build_ncf_model(num_students, num_topics)
ncf_model.fit([X_ncf_student_train, X_ncf_topic_train], y_ncf_train, epochs=10, batch_size=32, validation_data=([X_ncf_student_test, X_ncf_topic_test], y_ncf_test))

# Build and train Sequence Model (LSTM/GRU)
sequence_model = build_sequence_model(X_seq_train.shape[1:])
sequence_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, validation_data=(X_seq_test, y_seq_test))

# Function to calculate student's overall performance and suggest resources
def get_student_performance(student_roll_number):
    # Get student data
    student_id = student_encoder.transform([student_roll_number])[0]
    
    # Map the encoded student ID back to the original student ID
    original_student_id = student_encoder.inverse_transform([student_id])[0]
    
    # Access performance data using the original student ID
    student_performance = performance_matrix.loc[original_student_id]
    
    # Calculate overall performance (average score across subjects)
    overall_performance = student_performance.mean()
    
    # Classify the student's performance
    if overall_performance < 0.02:
        performance_status = 'Weak'
        # Suggest resources across all topics, not just performance topics
        resources = resources_data.head(5)  # Suggest top 5 resources for weak performance
    elif 0.02 <= overall_performance < 0.5:
        performance_status = 'Average'
        resources = resources_data.head(3)  # Suggest 3 resources for average performance
    else:
        performance_status = 'Good'
        resources = resources_data.head(1)  # Suggest 1 resource for good performance
    
    # Visualize student performance
    student_performance.plot(kind='bar', figsize=(10, 6), title=f'{student_roll_number} - Subject-wise Performance')
    plt.ylabel('Score')
    plt.xlabel('Subjects')
    plt.show()

    return overall_performance, performance_status, resources

# Example use case
student_roll_number = 'Student_361'  # Input roll number
overall_performance, performance_status, resources = get_student_performance(student_roll_number)

# Output
print(f"Overall Performance for {student_roll_number}: {overall_performance:.2f}")
print(f"Performance Status: {performance_status}")
print("Suggested Resources:")
print(resources[['Resource_ID', 'Topic', 'Platform', 'Type', 'Resource_Link']])
