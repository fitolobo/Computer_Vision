import csv
import tweepy
import config
import dataset
import pandas as pd
from asyncio import get_event_loop, sleep





def get_all_tweets(screen_name, writer):

    """Given a screen_name twitter extracts all the tweets."""
    auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_key, config.access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True,
                     retry_count=config.retry_count, retry_delay=config.delay)

    recenttweets = []
    recent_tweets = api.user_timeline(screen_name=screen_name, count=config.tweet_max_count, include_rts=False)
    recenttweets.extend(recent_tweets)

    Resultingtweets = [[tweet.user.name,
                        tweet.user.id,
                        tweet.user.screen_name,
                        tweet.id_str,
                        tweet.created_at,
                        tweet.text.encode("utf-8").decode("utf-8"),
                        tweet.favorite_count,
                        tweet.retweet_count] for tweet in recenttweets]

    writer.writerows(Resultingtweets)


async def all_twitter_extraction():

    with open('data/all_tweets.csv', 'w', encoding='utf-8') as f_all:

        writer = csv.writer(f_all)

        for screen_name_account in dataset.lista_cuentas_twitter:

            try:
                get_all_tweets(screen_name_account, writer)
                print("Done: ", screen_name_account)
            except tweepy.TweepError:
                print("Failed to run the command on that user, Skipping...", screen_name_account)
                pass

    names_col = ["User Name", "User_ID", "account_name", "tweet_id", "created_at", "Tweet_text", "favorite_count",
                 "retweet_count"]

    df_twitter = pd.read_csv("data/all_tweets.csv", names=names_col, index_col=False)

    df_twitter =  df_twitter.drop_duplicates(subset=["Tweet_text"])

    df_final = pd.merge( df_twitter, dataset.accounts_df, on="account_name", left_index=True)
    newdf = df_final.loc[(df_final.retweet_count >= 3)]
    newdf.to_csv('data/all_tweets_selected.csv', index=False)



async def run_function_every(time: int):
    """Run some code in interval given a lapse in seconds.
    Args:
        time (int): time in seconds to wait to run code
    """
    while True:

        await all_twitter_extraction()
        await sleep(time)


if __name__ == "__main__":
    # a loop is a manager that help us to
    # execute code we programmed to run
    # across time
    loop = get_event_loop()
    # loop.create_task tells loop to run a function
    # if you want to run more functions, put those
    # here with loop.create_task(function(arguments))
    loop.create_task(run_function_every(config.execution_time))
    # run loop and tasks created
    loop.run_forever()

