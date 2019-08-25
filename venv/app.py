
import africastalking

# Initialize SDK
username = "sandbox"    # use 'sandbox' for development in the test environment
api_key = "a14b15e85a02053f5c1ee092bb23af5ad2d320733478e7f0f330728f2573ba21"      # use your sandbox app API key for development in the test environment
africastalking.initialize(username, api_key)

# Initialize a service e.g. SMS
sms = africastalking.SMS
# Use the service synchronously
result = sms.send('Hello Sync Test', ['+254718769882'])
print('\nSync Done with -> ' + result['SMSMessageData']['Message'])

