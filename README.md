# Student Performance-Based Recommender System

## Project Overview
This project aims to develop a **Student Performance-Based Recommender System** that analyzes quiz performance data to provide personalized study resource recommendations. The system identifies weak areas, tracks trends, and suggests targeted resources, such as study materials and quizzes, to improve student learning outcomes. It integrates techniques like **collaborative filtering** and **clustering** to provide actionable and tailored insights.



## Features
1. **Personalized Recommendations**:
   - Provides students with study materials and quizzes tailored to their performance.
   - Uses **Neural Collaborative Filtering (NCF)** to predict quiz relevance.

2. **Performance Analysis**:
   - Clusters students into three categories: Weak, Average, and Good.
   - Uses **K-Means Clustering** based on average quiz scores.

3. **Dynamic Recommendation System**:
   - Recommends resources based on performance levels:
     - **Weak**: 5-6 resources per topic.
     - **Average**: 4-5 resources per topic.
     - **Good**: 2-3 resources per topic.

4. **Web-Based Interface**:
   - Allows users to input a **Student ID** to get recommendations.
   - Displays performance level and average score with suggested study resources.



## Technologies Used
### Data Analysis & Modeling:
- **Python**
- **Pandas** for data manipulation.
- **NumPy** for numerical operations.
- **Scikit-learn** for clustering and preprocessing.
- **Keras** for the deep learning model.

### Backend Development:
- **Flask** for creating RESTful APIs and hosting the application.
- **Flask-CORS** for enabling cross-origin requests.

### Frontend Development:
- **HTML**, **CSS**, and **JavaScript** for the web interface.



## How It Works
1. **Data Preprocessing**:
   - Encodes categorical fields (e.g., Student IDs, Quiz IDs, Topics).
   - Normalizes numerical fields (e.g., Scores, Time Taken).

2. **Clustering Students**:
   - Computes average scores for each student.
   - Clusters students into **Weak**, **Average**, and **Good** groups using **K-Means Clustering**.

3. **Collaborative Filtering**:
   - Builds a **Neural Collaborative Filtering (NCF)** model to predict quiz relevance.
   - Embeds student and quiz information into a latent space for similarity matching.

4. **Resource Recommendation**:
   - Recommends resources based on the student's performance level.
   - Allocates fewer resources to good performers and more resources to weaker students.

5. **Web Interface**:
   - Takes Student ID as input.
   - Displays performance level, average score, and recommended resources.



## Project Setup
1. **Clone the Repository**:
   bash
   git clone https://github.com/iqra077/StudentPerformance_Based_RecommenderSystem.git
   
   cd StudentPerformance_Based_RecommenderSystem
   

3. **Install Dependencies**:
   pip install -r requirements.txt

4. **Prepare Data**:
   - Place the required datasets in the project directory:
     - "student_performance_data.csv"
     - "study_resources_data.csv"

5. **Run the Application**:
   bash
   python test2.py
   

6. **Access the Application**:
   - Open your browser and visit: `http://127.0.0.1:5005



## File Structure

├── test2.py                   # Flask application

├── student_performance_data.csv

├── study_resources_data.csv

├── templates

│   └── index.html           # Frontend HTML

├── static

│   ├── styles.css           # CSS for frontend

│   └── scripts.js           # JavaScript for interactivity

├── requirements.txt         # Python dependencies

└── README.md                # Project documentation



## Contributing
Contributions are welcome! If you have suggestions for improvements, feel free to open an issue or submit a pull request.



