from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quiz.db'
    
    db.init_app(app)

    # Blueprint'leri burada import edin
    from .routes.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
