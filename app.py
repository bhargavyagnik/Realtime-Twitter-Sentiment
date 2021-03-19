from flask import Flask,render_template,request
from flask import Flask,render_template
import json
import twitterfiles
import tweepy
from textblob import TextBlob
# import os
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import numpy as np
app = Flask(__name__)
print("\n***Loading Model...\n")
model = tf.keras.models.load_model('model/zomato_sentiment101.h5',custom_objects={'KerasLayer': hub.KerasLayer})
print("\n***Model Loaded successfully...\n")

def gettweets():
    print("\n*** Fetching Tweets using tweepy library...\n")
    search_words = ["#zomatodeliveryboy", "#mentoo", "#SupportKamraj", "#supportzomatoguy", "#zomatocase","#reinstatekamraj"]
    auth = tweepy.OAuthHandler(twitterfiles.consumer_key, twitterfiles.consumer_secret)
    auth.set_access_token(twitterfiles.access_token, twitterfiles.access_token_secret)
    api = tweepy.API(auth)
    l = []
    date_since ="2021-03-17"
    for i in search_words:
        tweets = tweepy.Cursor(api.search,
                               q=i,
                               lang="en",
                               since=date_since).items(10)
        temp = [tweet.text for tweet in tweets]
        l.extend(temp)
    return l

def get_polarity(pred):
  x = np.argmax(pred[0])
  if x==0:
    return (-1*pred[0][0])
  elif x==1:
    return (0)
  else:
    return (1*pred[0][2])

def predict(tweets):
    l=[]
    for i in tweets:
        sent = twitterfiles.preprocessing(str(i))
        l.append(round(get_polarity(model.predict([sent])),2))
    return l


# def predict(tweets):
#     l=[]
#     for i in tweets:
#         sent = twitterfiles.preprocessing(str(i))
#         l.append(round(TextBlob(sent).sentiment.polarity,2))
#     return l


tweets=gettweets()
# tweets =["#HiteshaChandranee releases statement | Says haven't left Bengaluru Urges people not to form opinions until probe… https://t.co/q6fzBlcaj0",
# 'RT @KrishanPandit02: #HiteshaChandranee one of best funny meme video guyz see this #ReinstateKamaraj #Kamaraj #zomatodeliveryboy #Femin…',
# 'RT @yaifoundations: .#ZomatoDeliveryBoyHe is the only earning man in his family &amp; he lost his job bec of false allegation &amp; hastiness by p…',
# "RT @akasshngupta: #ReinstateKamaraj!The guy lost his job &amp; pride due to an arrogant influencer who seemed to be begging for 'free' food fo…",
# 'RT @gaur_vips: #rippedjeans is an attempt of feminists to divert the attention from #zomatodeliveryboy']
predictions = predict(tweets)
print(len(tweets))

@app.route('/')
def hello_world():
    return render_template('index.html',len=len(tweets),tweet=tweets,preds = predictions)

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        form_data = request.form
        tex=form_data.to_dict()['Name']
        pred=predict([tex])
        return render_template('data.html',pr= pred[0],text=tex,len=len(tweets),tweet=tweets,preds = predictions)

if __name__ == '__main__':
    app.run(debug=True)
