from flask import Blueprint, render_template, request, redirect, url_for, session
from .. import db
from ..models import Topic, TestQuestion, OpenEndedQuestion, Score, Question

bp = Blueprint('routes', __name__)

@bp.route('/')
def index():
    topics = Topic.query.all()
    return render_template('index.html', topics=topics)

@bp.route('/quiz/<int:topic_id>')
def quiz(topic_id):
    topic_name = Topic.query.get(topic_id).name
    test_questions = TestQuestion.query.filter_by(topic_id=topic_id).all()
    open_ended_questions = OpenEndedQuestion.query.filter_by(topic_id=topic_id).all()
    return render_template('quiz.html', test_questions=test_questions,topic_name=topic_name, open_ended_questions=open_ended_questions, topic_id=topic_id, best_score=get_best_score(topic_id))

@bp.route('/result/<int:topic_id>', methods=['POST'])
def result(topic_id):
    questions = Question.query.filter_by(topic_id=topic_id).all()
    score = 0
    for question in questions:
        user_answer = request.form.get(f'question_{question.id}')
        if not user_answer:
            continue
        if user_answer.upper() == question.answer.upper():
            score += 1
    session['score'] = score
   
    db.session.add(Score(score=score, topic_id=topic_id))
    db.session.commit()
    return redirect(url_for('routes.show_result', topic_id=topic_id))

@bp.route('/result/<int:topic_id>')
def show_result(topic_id):
    score = session.get('score', 0)
    return render_template('result.html', score=score, best_score=get_best_score(topic_id))

@bp.route('/add_question', methods=['POST'])
def add_question():
    question = request.form.get('question')
    option1 = request.form.get('option1')
    option2 = request.form.get('option2')
    option3 = request.form.get('option3')
    option4 = request.form.get('option4')
    answer = request.form.get('answer')

    new_question = Question(question=question, option1=option1, option2=option2, option3=option3, option4=option4, answer=answer)
    db.session.add(new_question)
    db.session.commit()

    return redirect(url_for('routes.quiz'))

def get_best_score(topic_id):
    score = Score.query.filter_by(topic_id=topic_id).order_by(Score.score.desc()).first()
    return score.score if score else 0
