from flask_api import FlaskAPI
from flask_sqlalchemy import SQLAlchemy
# local import
from instance.config import app_config
from flask import request, jsonify, abort, make_response

# initialize sql-alchemy
db = SQLAlchemy()

def create_app(config_name):
    from app.models import Company, User
    app = FlaskAPI(__name__, instance_relative_config=True)
    app.config.from_object(app_config[config_name])
    app.config.from_pyfile('config.py')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)



    @app.route('/companies/', methods=['POST','GET'])
    def companies():
        # Get the access token from the header
        auth_header = request.headers.get('Authorization')
        access_token = auth_header.split(" ")[1]

        if access_token:
            user_id = User.decode_token(access_token)
            if not isinstance(user_id, str):

                if request.method == "POST":
                    name = str(request.data.get('name', ''))
                    if name:
                        company = Company(name= name)
                        company.save()
                        response = jsonify({
                            'id':company.id,
                            'date_created':company.date_created,
                            'name':company.name,
                            'date_modified':company.date_modified
                        })
                        response.status_code =201
                        return response
                    else:
                        #GET
                        companies =Company.get_all()
                        results = []

                        for company in companies:
                            obj = {
                                'id':company.id,
                                'name':company.name,
                                'date_created':company.date_created,
                                'date_modified': company.date_modified
                            }  
                            results.append(obj)
                        response = jsonify(results)
                        response.status_code = 200
            
                        return response
            else:
                # user is not legit, so the payload is an error message
                message = user_id
                response = {
                    'message': message
                }
                return make_response(jsonify(response)), 401

    @app.route('/companies/<int:id>', methods=['GET','PUT','DELETE'])
    def companies_manipulation(id, **kwargs):
        # retrieve a campany using it's ID
        company = Company.query.filter_by(id=id).first()
        if not company:
            # Raise an HTTPException with a 404 not found status code
            abort(404)

        if request.method == 'DELETE':
            company.delete()
            return {
            "message": "company {} deleted successfully".format(company.id) 
         }, 200

        elif request.method == 'PUT':
            name = str(request.data.get('name', ''))
            company.name = name
            company.save()
            response = jsonify({
                'id': company.id,
                'name': company.name,
                'date_created': company.date_created,
                'date_modified': company.date_modified
            })
            response.status_code = 200
            return response
        else:
            # GET
            response = jsonify({
                'id': company.id,
                'name': company.name,
                'date_created': company.date_created,
                'date_modified': company.date_modified
            })
            response.status_code = 200
            return response

    # import the authentication blueprint and register it on the app
    from .auth import auth_blueprint
    app.register_blueprint(auth_blueprint)


    return app