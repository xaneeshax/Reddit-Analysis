# In[1]:


import praw
import tweepy
import requests
from alpha_vantage.timeseries import TimeSeries
import pandas_datareader.data as web

import pandas as pd
from collections import Counter

from datetime import datetime, timedelta
import time
import math

import wordcloud as wc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cufflinks as cf

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as plyo
plyo.init_notebook_mode(connected=True)


# ### Scraping Reddit 

# In[2]:
# Function to access Reddit PRAW

def get_reddit(username, password):
    """ Accesses the Reddit API """
    
    return praw.Reddit(client_id = 'kRYLDf*******',
                       client_secret= 'wJysO3z6P************',
                       username = username,
                       password = password,
                       user_agent = 'dsproject')


# In[3]:
# Calling the get_reddit function

reddit = get_reddit('username', 'password')


# In[4]:
    
def scrape_reddit(subreddit_name, num_posts):
    """ Scrapes a Reddit Thread given a subreddit name and number of posts """
    
    df = pd.DataFrame()
    
    subreddit = reddit.subreddit()
    for submission in subreddit.top(limit=num_posts):
    
        comments = []
        submission.comments.replace_more(limit = 0)
        
        for comment in submission.comments.list():
            comments.append(comment.body)
            
        df = df.append({'created_utc' : submission.created_utc,
                      'comments' : comments,
                      'score' : submission.score,
                      'title' : submission.title,
                      'upvote_ratio' : submission.upvote_ratio}, 
                       ignore_index = True)

    df.to_csv(subreddit_name + '.csv')


# In[5]:
# Scraping the r/wallstreetbets subreddit

subreddit_name = "wallstreetbets"
scrape_reddit(subreddit_name, 800)
bets = pd.read_csv(subreddit_name + '.csv')


# In[6]:
# Dictionary for converting month to date

month_date = {
    'Jan' : 1,
    'Feb' : 2,
    'Mar' : 3,
    'Apr' : 4,
    'May' : 5,
    'Jun' : 6,
    'Jul' : 7,
    'Aug' : 8,
    'Sep' : 9,
    'Oct' : 10,
    'Nov' : 11,
    'Dec' : 12,
}


# In[7]:
# Function to convert month to date

def format_month(month):
    return month_date[month]


# In[8]:
# Dictionary for converting date to month
date_month = {}

for key, val in month_date.items():
    date_month[val] = key


# In[9]:
# Removes the extra 0 in dates such as 2020-01-01

def format_date(date):
    if date[0] == '0':
        date = int(date[1:])
    else:
        date = int(date)
    
    return date


# In[10]:
# Splits a time in string format to hours, mins, and secs

def split_time(time):
    hours, mins, secs = time.split(':')
    return int(hours), format_date(mins), format_date(secs) 


# In[11]:
# Splits a time in string format and gets the minutes

def get_mins(time):
    hours, mins, secs = time.split(':')
    return format_date(mins) 


# In[12]:
# Formats the month in a given date

def get_month(month):
    return format_date(month)


# In[13]:
# Takes time in UTC format and outputs the Datetime Format in EST

def date(utc):
    info = time.strftime("%a, %d %b %Y %H:%M:%S %Z", 
                         time.localtime(utc)).split()
    hours, mins, secs = split_time(info[4])
    
    created = datetime(int(info[3]), format_month(info[2]), 
                       format_date(info[1]), hours, mins, secs)
    created_local = created.strftime('%Y-%m-%dT%H:%M:%S')
    created_est = (created + timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')
    
    return created_est


# In[14]:
# Cleaning/Reformatting the Dataframe

# Sorts the Dataframe using the UTC time
bets = bets.sort_values('created_utc').reset_index()

# Date column represents Datetime format
bets['Date'] = bets.apply(lambda row: date(row['created_utc']), axis = 1)

# Year Column parses the date and stores the year
bets['Year'] = bets.apply(lambda row: int(row['Date'][0:4]), axis = 1)

# Only keep 2020 data
bets = bets[bets['Year'] == 2020]

# Keep the Month column
bets['Month'] = bets.apply(lambda row: get_month(row['Date'][5:7]), axis = 1)

# The Date column
bets['Day'] = bets.apply(lambda row: row['Date'][:10], axis = 1)

# Reset Index
bets = bets.drop(['index', 'Unnamed: 0'], axis=1)


# In[15]:
# Cleaning Comment Data

# Opens the stopwords file to assist cleaning the comment and post data
filename = open('stopwords.txt', 'r') 
stopwords = filename.readlines()

for i in range(len(stopwords)):
    try:
        stopwords[i] = stopwords[i][: stopwords[i].index('\n')]
    except:
        stopwords[i] = stopwords[i]


# In[16]:

def replace_punctuation(comments):
    """Removes all punctuation, emojis, and links from the data"""
    
    # Splits the words into a list of words
    words = comments.split()
    
    # Removes all punctuation, links, emjois etc.
    for i in range(len(words)):
        new = ''
        for character in words[i]:
            if character.isalnum():
                new += character.lower()
        words[i] = new
    
    # Filters non-essential words     
    return [word for word in words if len(word) > 2 and word != '' and word not in stopwords]


# In[17]:
# Generates wordclouds for 2020

months = {}

for month, comment in zip(bets.Month, bets.comments):
    months[month] = months.get(month, []) + replace_punctuation(comment) 


# In[18]:
# Creating a wordcloud object

cloud = wc.WordCloud(width=600, height=600,
                    colormap = 'Dark2',
                    background_color = 'white')


# In[19]:
# Plots the wordcloud data

fig = plt.figure(figsize = (12,12))
columns = 4
rows = 3

for i in range(1, columns * rows + 1):
    img = cloud.generate(' '.join(months[i]))
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.title(date_month[i])
    plt.imshow(img)


# In[20]:
# Function to plot Monthly Wordclouds

# Generates word clouds given a month
def month_wc(month_name):
    
    # Takes every 4 posts in a give month
    month_sentiment = bets[bets['Month'] == month_date[month_name]][::4]
    
    # Stores the date & time and the post data
    posts = {}
    for date, comment in zip(month_sentiment.Date, month_sentiment.comments):
        posts[date.replace('T', ' ')] = replace_punctuation(comment)
        
    dates = list(posts.keys())
    
    # Generates word clouds for each key-value pair
    fig = plt.figure(figsize = (12,12))
    columns = 4
    rows = 4

    for i in range(1, columns * rows + 1):
        try:
            img = cloud.generate(' '.join(posts[dates[i-1]]))
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.title(dates[i-1])
            plt.imshow(img)
        except:
            break


# In[21]:
# Calling the function on data from January

month_wc('Jan')


# In[22]:
# Stores the cleaned comments in the dataframe

bets['comments_edited'] = bets.apply(lambda row: replace_punctuation(row['comments']), axis = 1)


# In[23]:
# Analyzes the number of calls and puts mentioned and the ratio

call_put = {}
cp_ratio = {}


# In[24]:
# Counts the number of times "Call" and "Put" is mentioned each day

for day, words in zip(bets.Day, bets.comments_edited):
    
    calls = 0
    puts = 0
    
    for word in words:
        if 'call' in word:
            calls += 1
        elif 'put' in word:
            puts += 1
            
    if calls > 0 or puts > 0:
        counts = call_put.get(day, [0,0])
        call_put[day] = [counts[0]+calls, counts[1]+puts]


# In[25]:
# Using the number of class & puts to calculate the call/put ratio

for date in call_put.keys():
    cp_ratio[date] = round(call_put[date][0] / call_put[date][1], 4)


# In[26]:
# Uses the data to create a dataframe

cprs = pd.DataFrame({'Date' : list(cp_ratio.keys()), 
                    'Ratios' : list(cp_ratio.values()),
                    'Puts' : [vals[1] for vals in call_put.values()],
                    'Calls' : [vals[0] for vals in call_put.values()] })


# In[27]:
# Getting stock data

def stock_data(ticker, start=dt.datetime(2020,1,1)):
    import datetime as dt
    end = dt.datetime.now()
    df = pd.DataFrame()
    df = web.DataReader(ticker, 'yahoo', start, end)
    return df


# In[28]:
# Read & Store SPY Data
spy_data = pd.read_csv('spy.csv')


# In[29]:
# Merging the Datasets

merged = pd.merge(cprs, spy_data, on='Date')
merged = merged.dropna()


# In[30]:
# Visualizing the merged data
merged.tail(3)


# In[31]:
# Visualizing the Calls vs Puts

# Remove Outliers
calls = [vals[0] for vals in call_put.values() if vals[0] < 600]

puts = [vals[1] for vals in call_put.values()]
puts.pop(212)

dates = [date for date in list(call_put.keys()) if date != '2020-09-16']


# In[32]:
# Plotting the # of "call"  mentions and the # of "put" mentions for
# each day of 2020

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=dates, y=calls, mode='markers',
                         name='Calls', marker_color='#2adea5'))

fig.add_trace(go.Scatter(x=dates, y=puts, mode='markers',
                         name='Puts', marker_color='#e8666d'))

fig.update_layout(title= 'Calls vs Puts Metions on r/wallstreetbets',
                  xaxis_title='Date',
                  yaxis_title='Times Mentioned',
                  legend_title= 'Option Type')
fig.show()


# In[33]:
# Plotting the # of "put" mentions and the Adj. Close of the SPY ETF

fig = px.scatter(merged, x="Puts", y="Adj Close", hover_name='Date', 
                 trendline="ols",
                 color_discrete_sequence = ['#de4d43'],
                 labels = {'Puts' : 'Number of \"Put\" Mentions',
                           'Adj Close' : 'Adjusted Close of SPY ETF'},
                title = 'Correlation between SPY ETF Closing Price and \"Put\" Mentions on r/wallstreetbets')
fig.show()


# In[34]:
# Remove outliers
 
merged_discard_outliers = merged[merged['Ratios'] < 9]

fig = px.scatter(merged_discard_outliers, x="Adj Close", y="Ratios", 
                 hover_name='Date', trendline="ols",
                 color_discrete_sequence = ['#4ecc95'],
                 labels = {'Ratios' : 'Ratio of Call to Put Mentions',
                           'Adj Close' : 'Adjusted Close of SPY ETF'},
                title = 'Correlation between SPY ETF Closing Price & Ratio of Call to Put Mentions on r/wallstreetbets')
fig.show()


# In[35]:
# Read & Store SPY Data

spy = pd.read_csv('spy.csv')
spy = spy.set_index('Date')
spy['Pct Change'] = spy['Adj Close'].astype(float).pct_change()


# In[36]:
# Read & Store VIX Data

stock_data('VIX').to_csv('vix.csv')
vix = pd.read_csv('vix.csv')
vix = vix.set_index('Date')


# In[37]:
# Plots the VIX ETF data

def plot(df):
    
    columns = ['High', 'Close', 'Low']
    
    plyo.iplot(
        df[columns].iplot(asFigure=True,
                          theme='polar',
                          title= '^VIX ETF',
                          xTitle='Date',
                          yTitle='Price per Share',
                          mode={'High': 'markers', 'Close' : 'lines+markers', 'Low': 'markers'},
                          symbol={'High': 'circle', 'Close' : 'circle', 'Low': 'circle'},
                          size=3.5,
                          colors={'High': '#5f12e3', 'Close' : '#983aab', 'Low': '#a783e6'}
                         ) 
 )


# In[38]:
# Plotting VIX 

plot(vix)


# In[39]:
# Plots the movement of the share price

def movement(df):
    quotes = df.loc[:, ['Open', 'High', 'Low', 'Close']]
    
    qf = cf.QuantFig(
                      quotes,
                      title='SPY ETF Data 2020',
                      legend='top',
                      name='Share Price'
             )

    plyo.iplot(
                 qf.iplot(asFigure=True)
             )


# In[40]:
# Plotting SPY

movement(spy)


# In[41]:
# Company Mentions and Stock Movement

# Get data for each company in the wordclouds and create a dataframe
stocks = pd.DataFrame({'MSFT' : stock_data('MSFT')['Adj Close'],
                      'AAPL' : stock_data('AAPL')['Adj Close'],
                      'TSLA' : stock_data('TSLA')['Adj Close'],
                      'NKLA' : stock_data('NKLA')['Adj Close'],
                      'PLTR' : stock_data('PLTR', datetime(2020,9,27))['Adj Close']})


# In[42]:
# Find the percent change and the absolute value of percent change

for name in ['MSFT', 'AAPL', 'TSLA', 'NKLA', 'PLTR']:
    stocks[name + '_pct'] = stocks[name].pct_change()
    stocks[name + '_pos_pct'] = stocks.apply(lambda row: abs(row[name + '_pct']), axis = 1)


# In[43]:
# String representation of the dates

dates = [str(date)[:10] for date in list(stocks.index)]


# In[44]:
# Company Frequencies

# Finds the number of times each company was mentioned in the posts
frequency = {date: [0,0,0,0,0] for date in dates}


# In[45]:
# Calculates the number of mentions for each company


MSFT, AAPL, TSLA, NKLA, PLTR = 0,1,2,3,4

for day, words in zip(bets.Day, bets.comments_edited):
    if day in frequency.keys():
        for word in words:
            if word in ['msft', 'microsoft']:
                frequency[day][MSFT] += 1
            elif word in ['aapl', 'apple']:
                frequency[day][AAPL] += 1
            elif word in ['tsla', 'tesla']:
                frequency[day][TSLA] += 1
            elif word in ['nkla', 'nikola']:
                frequency[day][NKLA] += 1
            elif word in ['pltr', 'palantir']:
                frequency[day][PLTR] += 1


# In[46]:
# Converts data into a dataframe

frequency_df = pd.DataFrame(frequency).T
frequency_df.columns = list(stocks.columns)[:5]


# In[47]:
# View frequency dataframe

frequency_df.tail(3)


# In[48]:
# Merge stock data with the frequency data

df_merged = stocks.merge(frequency_df, how='inner', left_index=True, right_index=True)


# In[49]:
# Visualize Frequency Data

# Scatterplot to visulize TESLA data
fig = go.Figure()

pcts = df_merged['TSLA_pos_pct'][125:]
pcts = [pct *100 for pct in pcts]

# Add traces
fig.add_trace(go.Scatter(x=pcts, y=df_merged['TSLA_y'][125:], mode='markers',
                         name='Calls', marker_color='#2adea5'))

fig.update_layout(title= 'Relationship between TSLA stock movement and Times Mentioned on Reddit',
                  xaxis_title='Absolute Value of Daily Percent Change',
                  yaxis_title='Times Mentioned')
fig.show()


# In[50]:
# Function to plot any company data aginst its frequency on Reddit

def plot_mentions_and_price_change(ticker, start_day=0):
    """Plots company's stock data vs mentions on Reddit (measures social media popularity)"""
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=dates[start_day:], y=df_merged[ticker + '_y'][start_day:], name='Subreddit Mentions'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dates[start_day:], y=df_merged[ticker + '_pos_pct'][start_day:], name = 'Share Price Movement'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text= f'{ticker} Subreddit Mentions vs Share Price Movement'
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="Number of mentions on Subreddit", secondary_y=False)
    fig.update_yaxes(title_text="Absolute Value of Percent Change", secondary_y=True)

    fig.show()


# In[51]:
# Calling the plot mentions funcion
plot_mentions_and_price_change('PLTR', 189)


# In[52]:
# Word Frequencies against SPY & VIX

# Function to plot the frequeny of a word on Reddit with ETF data
def word_frequencies_in_2020(target_words, title):
    
    common = {}
    
    # Determines on which days any of the target words were used and how many times they occured
    for day, words in zip(bets.Day, bets.comments_edited):
        for word in words:
            if word in target_words:
                common[day] = common.get(day, 0) + 1
    
    # Normalizing the dataset so the range of all the values is in between 0 and 1
    stock_vals = []
    max_val = max(list(common.values()))
    for val in list(common.values()):
        stock_vals.append(val/ max_val)
    
    # Plots the SPY data, VIX data, and Frequency of the occurence of a word
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = dates, y=list(spy['Adj Close'] / max(spy['Adj Close'])), mode='lines', 
                             name = 'SPY', marker_color = '#37cfed'))
    
    fig.add_trace(go.Scatter(x = dates, y=list(vix['Adj Close'] / max(vix['Adj Close'])), mode='lines', 
                             name = '^VIX', marker_color = '#e33947'))
    
    fig.add_trace(go.Scatter(x = list(common.keys()), y=stock_vals, mode='lines', 
                             name='Keyword Frequency', marker_color = '#2fde26'))

    fig.update_layout(title= f'Frequency of words related to {title}',
                      xaxis_title='Days', yaxis_title='Percentage of its High')
    fig.show()
    


# In[53]:
# Calling the word frequencies function

word_frequencies_in_2020(['hold', 'holding'], 'Holds')


# In[54]:
# Top Words - Daily

# Shows the Top 10 words used for each day
def top_words_daily(date):
    
    all_words = {}

    for day, words in zip(bets.Day, bets.comments_edited):
        if day == date:
            for word in words:
                if len(word) > 3:
                    all_words[word] = all_words.get(word, 0) + 1

    frequency = Counter(all_words).most_common()[:10]

    words = [vals[0] for vals in frequency]
    freq = [vals[1] for vals in frequency]

    fig = go.Figure([go.Bar(x=words, y=freq)])

    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text= f'Top Words used on r/wallstreetbets - {date}')

    fig.show()


# In[55]:
# Calls the Top Words function
top_words_daily('2020-03-07')





