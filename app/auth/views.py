from . import auth_blueprint

from flask.views import MethodView
from flask import make_response, request, jsonify
from app.models import User



class RegistrationView(MethodView):
    """This class registers a new user."""

    def post(self):
        """Handle POST request for this view. Url ---> /auth/register"""

        # Query to see if the user already exists
        user = User.query.filter_by(email=request.data['email']).first()

        if not user:
            # There is no user so we'll try to register them

            try:
                post_data = request.data
                # Register the user
                print("user is", post_data)
                name = post_data['business_name']
                email = post_data['email']
                password = post_data['password']
                user = User(name= name, email=email, password=password)
                user.save()
                token = User.generate_token(self,user.id)
                response = {
                    'status': 'success',
                    'token': token.decode()
                }



                # return a response notifying the user that they registered successfully
                return make_response(jsonify(response)), 201
            except Exception as e:
                # An error occured, therefore return a string message containing the error
                response = {
                    'message': str(e)
                }
                return make_response(jsonify(response)), 401
        else:
            # There is an existing user. We don't want to register users twice
            # Return a message to the user telling them that they they already exist
            response = {
                'message': 'User already exists. Please login.'
            }

            return make_response(jsonify(response)), 202





class UserView(MethodView):
    def get(self):
        """This endpoints gets user info and returns"""

        print('/auth/user reached')





class LoginView(MethodView):
    """This class-based view handles user login and access token generation."""

    def post(self):
        """Handle POST request for this view. Url ---> /auth/login"""
        print ('incoming request',request.data )
        try:
            # Get the user object using their email (unique to every user)
            user = User.query.filter_by(email=request.data['email']).first()

            # Try to authenticate the found user using their password
            if user and user.password_is_valid(request.data['password']):
                # Generate the access token. This will be used as the authorization header

                token = User.generate_token(self,user.id)

                if token:

                    response = {
                        'status': 'success',
                        'token': token.decode(),
                        'user':user.email,
                        'business':user.name
                    }

                    return make_response(jsonify(response)), 200
            else:
                # User does not exist. Therefore, we return an error message
                response = {
                    'message': 'Invalid email or password, Please try again'
                }
                return make_response(jsonify(response)), 401

        except Exception as e:
            # Create a response containing an string error message
            print("error detected")
            response = {
                'message': str(e)
            }
            # Return a server error using the HTTP Error Code 500 (Internal Server Error)
            return make_response(jsonify(response)), 500

registration_view = RegistrationView.as_view('register_view')
login_view = LoginView.as_view('login_view')
user_view = UserView.as_view('user_view')


# Define the rule for the registration url --->  /auth/register
# Then add the rule to the blueprint
auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST','GET'])

auth_blueprint.add_url_rule(
    '/auth/user',
    view_func=user_view,
    methods=['GET']
)
# Define the rule for the registration url --->  /auth/login
# Then add the rule to the blueprint
auth_blueprint.add_url_rule(
    '/auth/login',
    view_func=login_view,
    methods=['POST']
)