#Import the necessary methods from tweepy library
import os
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


#This is a basic listener that just prints received tweets to stdout.
class Streaming(StreamListener):
    print("auth", access_token)
    def on_data(self, data):

        # print(os.path)
        with open('/home/dennis/Desktop/projects/python/projectAPI/app/analysis/sream_data.txt', 'w') as f:
            f.write(data)

        return True

    def on_error(self, status):
        print (status)

class GetTweets(MethodView):
    def post(self):
        """Handle POST request for this view. Url ---> /stream"""
        print ('incoming request',request.data )

        #This handles Twitter authetification and the connection to Twitter Streaming API
        l = Streaming()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        stream = Stream(auth, l)

        #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
        stream.filter(track=request.data['query'],languages=["en"])



streaming_view = GetTweets.as_view('stream_view')

# Define the rule for the streaming url --->  /stream
# Then add the rule to the blueprint
analysis_blueprint.add_url_rule(
    '/stream',
    view_func=streaming_view,
    methods=['POST'])