from app import create_app, db
from app.models import Topic, TestQuestion, OpenEndedQuestion, Score, Question
from app.routes import routes

app = create_app()

def add_sample_topics():
    sample_topics = [
        Topic(name='AI development in Python'),
        Topic(name='Computer vision'),
        Topic(name='NLP (Neuro-linguistics)'),
        Topic(name='Implementing AI models in Python applications'),
    ]
    
    db.session.bulk_save_objects(sample_topics)
    db.session.commit()

def add_sample_ai_questions():
    ai_topic_id = Topic.query.filter_by(name='AI development in Python').first().id
    ai_questions = [
    TestQuestion(question='Which Python library is commonly used for deep learning and neural network models?', 
                 option1='Pandas', 
                 option2='TensorFlow', 
                 option3='Scikit-learn', 
                 option4='Matplotlib', 
                 answer='TensorFlow', 
                 topic_id=ai_topic_id),
    TestQuestion(question='What method is used to evaluate the performance of a classification model in Python\'s Scikit-learn library?', 
                 option1='fit()', 
                 option2='predict()', 
                 option3='score()', 
                 option4='transform()', 
                 answer='score()', 
                 topic_id=ai_topic_id),
    TestQuestion(question='Which function from the pandas library is used to read a CSV file into a DataFrame?', 
                 option1='read_csv()', 
                 option2='import_csv()', 
                 option3='load_csv()', 
                 option4='parse_csv()', 
                 answer='read_csv()', 
                 topic_id=ai_topic_id),
    OpenEndedQuestion(question='What term describes the process of tuning hyperparameters to improve model performance?', 
                      answer='OPTIMIZATION', 
                      topic_id=ai_topic_id),
    OpenEndedQuestion(question='What is the name of the technique used to prevent overfitting by randomly dropping units during training?', 
                      answer='DROPOUT', 
                      topic_id=ai_topic_id),
]

    db.session.bulk_save_objects(ai_questions)
    db.session.commit()

def add_sample_cv_questions():
    cv_topic_id = Topic.query.filter_by(name='Computer vision').first().id
    cv_questions = [
    TestQuestion(question='Which Python library is widely used for computer vision tasks and provides functions for image processing?', 
                 option1='TensorFlow', 
                 option2='OpenCV', 
                 option3='Pandas', 
                 option4='Scikit-learn', 
                 answer='OpenCV', 
                 topic_id=cv_topic_id),
    TestQuestion(question='What function in OpenCV is used to convert a color image to grayscale?', 
                 option1='cv2.resize()', 
                 option2='cv2.cvtColor()', 
                 option3='cv2.threshold()', 
                 option4='cv2.flip()', 
                 answer='cv2.cvtColor()', 
                 topic_id=cv_topic_id),
    TestQuestion(question='In the context of image processing, what is the purpose of applying a Gaussian blur to an image?', 
                 option1='Increase contrast', 
                 option2='Reduce noise', 
                 option3='Detect edges', 
                 option4='Enhance colors', 
                 answer='Reduce noise', 
                 topic_id=cv_topic_id),
    OpenEndedQuestion(question='What technique is used to detect and track objects in video sequences?', 
                      answer='OBJECT TRACKING', 
                      topic_id=cv_topic_id),
    OpenEndedQuestion(question='What is the term for the process of identifying and labeling objects in an image using machine learning algorithms?', 
                      answer='OBJECT DETECTION', 
                      topic_id=cv_topic_id),
]

    db.session.bulk_save_objects(cv_questions)
    db.session.commit()

def add_sample_nlp_questions():
    nlp_topic_id = Topic.query.filter_by(name='NLP (Neuro-linguistics)').first().id
    nlp_questions = [
    TestQuestion(question='Which Python library is commonly used for natural language processing tasks and includes features for tokenization and text analysis?', 
                 option1='NumPy', 
                 option2='NLTK', 
                 option3='Matplotlib', 
                 option4='Seaborn', 
                 answer='NLTK', 
                 topic_id=nlp_topic_id),
    TestQuestion(question='What is the purpose of the `n-gram` model in natural language processing?', 
                 option1='Generate word embeddings', 
                 option2='Analyze sentence sentiment', 
                 option3='Predict the next word based on the previous n-1 words', 
                 option4='Translate text into another language', 
                 answer='Predict the next word based on the previous n-1 words', 
                 topic_id=nlp_topic_id),
    TestQuestion(question='Which method in the spaCy library is used to lemmatize words?', 
                 option1='spacy.tokenizer()', 
                 option2='spacy.nlp()', 
                 option3='spacy.lemmatizer()', 
                 option4='spacy.tokenize()', 
                 answer='spacy.lemmatizer()', 
                 topic_id=nlp_topic_id),
    OpenEndedQuestion(question='What term describes the process of converting text into numerical vectors for use in machine learning models?', 
                      answer='VECTORIZATION', 
                      topic_id=nlp_topic_id),
    OpenEndedQuestion(question='What is the name of the technique used to identify and extract named entities (such as people, organizations, locations) from text?', 
                      answer='NAMED ENTITY RECOGNITION', 
                      topic_id=nlp_topic_id),
]

    db.session.bulk_save_objects(nlp_questions)
    db.session.commit()

def add_sample_ai_app_questions():
    ai_app_topic_id = Topic.query.filter_by(name='Implementing AI models in Python applications').first().id
    ai_app_questions = [
    TestQuestion(question='Which Python library provides a simple interface for deploying machine learning models and integrates with various deployment platforms?', 
                 option1='TensorFlow', 
                 option2='Keras', 
                 option3='Flask', 
                 option4='Scikit-learn', 
                 answer='Flask', 
                 topic_id=ai_app_topic_id),
    TestQuestion(question='What is the main advantage of using a pre-trained model in Python for a new machine learning task?', 
                 option1='Reduces the need for large datasets', 
                 option2='Provides more complex algorithms', 
                 option3='Increases computational requirements', 
                 option4='Requires more programming knowledge', 
                 answer='Reduces the need for large datasets', 
                 topic_id=ai_app_topic_id),
    TestQuestion(question='In the context of deploying AI models with Flask, what function is typically used to handle HTTP POST requests containing model input data?', 
                 option1='app.get()', 
                 option2='app.post()', 
                 option3='app.route()', 
                 option4='app.request()', 
                 answer='app.post()', 
                 topic_id=ai_app_topic_id),
    OpenEndedQuestion(question='What technique is used to fine-tune a pre-trained model on a specific task or dataset?', 
                      answer='TRANSFER LEARNING', 
                      topic_id=ai_app_topic_id),
    OpenEndedQuestion(question='What is the name of the process that involves converting a machine learning model into a format suitable for production deployment?', 
                      answer='MODEL SERIALIZATION', 
                      topic_id=ai_app_topic_id),
]

    db.session.bulk_save_objects(ai_app_questions)
    db.session.commit()

def add_initial_scores():
    topics = Topic.query.all()
    for topic in topics:
        db.session.add(Score(score=0, topic_id=topic.id))
    db.session.commit()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if Topic.query.count() <= 0:
            add_sample_topics()
        if Question.query.count() <= 0:
            add_sample_ai_questions()
            add_sample_cv_questions()
            add_sample_nlp_questions()
            add_sample_ai_app_questions()
        if Score.query.count() <= 0:
            add_initial_scores()

    app.run(debug=False)

    
