from flask import Blueprint

# This instance of a Blueprint that represents the authentication blueprint
analysis_blueprint = Blueprint('analysis', __name__)

from . import streaming