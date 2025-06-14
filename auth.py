import os
from flask import Blueprint, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint('auth', __name__)

login_manager = LoginManager()
oauth = OAuth()

class User(UserMixin):
    def __init__(self, user_id, email, name):
        self.id = user_id
        self.email = email
        self.name = name

    @staticmethod
    def get(user_id):
        # In a real app, you'd query a database.
        # Here, we'll just reconstruct the user from the session.
        user_json = session.get('user')
        if user_json and user_json.get('id') == user_id:
            return User(user_id=user_json['id'], email=user_json.get('email'), name=user_json.get('name'))
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

def init_auth(app):
    login_manager.init_app(app)
    oauth.init_app(app)

    oauth.register(
        name='google',
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

@auth_bp.route('/login')
def login():
    redirect_uri = url_for('auth.authorize', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@auth_bp.route('/auth/google/callback')
def authorize():
    token = oauth.google.authorize_access_token()
    user_info = token['userinfo']
    
    # Create a user object
    user = User(
        user_id=user_info['sub'], 
        email=user_info['email'],
        name=user_info.get('name')
    )
    
    # Store user in session for Flask-Login
    session['user'] = {'id': user.id, 'email': user.email, 'name': user.name}
    login_user(user)
    
    return redirect(url_for('main.index'))

@auth_bp.route('/logout')
def logout():
    logout_user()
    session.pop('user', None)
    return redirect(url_for('main.index')) 