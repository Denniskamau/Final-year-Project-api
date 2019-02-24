#Import the necessary methods from tweepy library
import os
import json
import asyncio
import tweepy
from .prediction import Prediction
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from . import analysis_blueprint
from flask.views import MethodView
from flask import make_response, request, jsonify
from flask_json import FlaskJSON, JsonError, json_response, as_json

#Variables that contains the user credentials to access Twitter API
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""


all_tweets = []
class GetTweets(MethodView):
    response = []
    def post(self):
        """Handle POST request for this view. Url ---> /stream"""
        print ('incoming request',request.data )

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True,retry_count=10,retry_delay=5,retry_errors=5)
        query = request.data['query']
        count = 0
        date_since = "2019-01-01"
        tweets = tweepy.Cursor(api.search,q=query,lang="en",since=date_since).items(10)

        for tweet in tweets:
            predict = Prediction()
            results = predict.receiveAnalysisParameter(tweet.text)
            self.response.append(results)

            count +=1
            if count >=10:
                print("count is greater than 10")

        return make_response(self.response)





streaming_view = GetTweets.as_view('stream_view')

# Define the rule for the streaming url --->  /stream
# Then add the rule to the blueprint
analysis_blueprint.add_url_rule(
    '/stream',
    view_func=streaming_view,
    methods=['POST'])