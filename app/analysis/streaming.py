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


#Variables that contains the user credentials to access Twitter API
access_token = "360352221-FJPX4r8ttCVWiQS9ZNBe8wCSruyGsq7mQxitXY1o"
access_token_secret = "rNm8eCtdJ6s8DN9dtYPgMLqe549PUqQdmCT3nVhaZ6Cbv"
consumer_key = "dmZVwBgGoE8hG27TLf3k1cUX6"
consumer_secret = "suKcD9XCcy5Fd2VdiPr0GS1sSt3MRFvJEckC9gbFCnrNefSevk"


all_tweets = []
class GetTweets(MethodView):
    def post(self):
        """Handle POST request for this view. Url ---> /stream"""
        print ('incoming request',request.data )
        response = []

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True,retry_count=10,retry_delay=5,retry_errors=5)
        query = request.data['query']
        count = 0
        date_since = "2019-01-01"
        tweets = tweepy.Cursor(api.search,q=query,lang="en",since=date_since).items(10)

        for tweet in tweets:

            print("tweet is",tweet.text)
            predict = Prediction()
            results = predict.receiveAnalysisParameter(tweet.text)
            response.append(results)
            #print("result is",results)
            count +=1
            if count >=10:
                print("count is greater than 10")
                #return make_response(jsonify(response)), 201
                print("all result",response)

                data = {
                    "message":"Finished anaysis of the first 10 tweets",
                    "status":"success",
                    "response":response
                }
                return jsonify(value=response)





streaming_view = GetTweets.as_view('stream_view')

# Define the rule for the streaming url --->  /stream
# Then add the rule to the blueprint
analysis_blueprint.add_url_rule(
    '/stream',
    view_func=streaming_view,
    methods=['POST'])