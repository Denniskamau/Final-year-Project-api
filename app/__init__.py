from flask_api import FlaskAPI
from flask_sqlalchemy import SQLAlchemy
# local import
from instance.config import app_config
from flask import request, jsonify, abort

# initialize sql-alchemy
db = SQLAlchemy()

def create_app(config_name):
    from app.models import Company
    app = FlaskAPI(__name__, instance_relative_config=True)
    app.config.from_object(app_config[config_name])
    app.config.from_pyfile('config.py')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)



    @app.route('/companies/', methods=['POST','GET'])
    def companies():
        if request.method == "POST":
            name = str(request.data.get('name', ''))
            if name:
                company = Company(name= name)
                company.save()
                response = jsonify({
                    'id':company.id,
                    'name':company.name,
                    'date_created':company.date_created,
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
    return app