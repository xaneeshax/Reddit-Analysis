# REDDIT ANALYSIS

# Introduction

The objective of this analysis is to find correlations between the movements of the stock market with the content on Reddit’s largest investment community, r/wallstreetbets. Data was collected using the Reddit and the Yahoo Finance APIs. The goals of this project included successfully scraping data through Reddit, using natural language processing to analyze the most frequent words, observing correlations between the number of times a company was mentioned and its price movement, determining the ratio of the number of times “call” and “puts” were used over the course of 2020, and using regression models to better analyze the dataset. The motivation and significance of this analysis was to better understand the social media response to financial gains and losses. Moreover, a goal was to determine if this data source could be used for predictive modelling as this data source is often used by quants [2]. The driving hypotheses for this analysis included that the Calls-to-Puts ratio would increase over the course of the year, that the most talked about companies this year would include Tesla and Apple, and that there is a negative correlation between the price of the SPY ETF and the number of times “Puts” was mentioned in the subreddit.


# Data Sources and Methods

I acquired my data using the Reddit’s PRAW (Python Reddit API Wrapper) [1]. I created an account and modified the settings in order to attain a client ID and a client secret code. I used these codes to access PRAW and create a reddit object. Using the documentation, I accessed the r/wallstreetbets subreddit and changed the settings to see all the top posts in 2020. For every submission I extracted the time created in UTC format, the comments on the post, the number of upvotes the post had, the tile of the post, and the ratio of upvotes to downvotes. To obtain the financial data I used the requests and pandas data reader module in order to read financial data for a given ETF or company for the specified timeframe. 
	In order to clean the data into a useable format I first had to convert UTC time into a Datetime format. In order to do so, I converted UTC time into a string format in the Pacific Time zone. I parsed the data using the datetime module in order to convert the value into a Datetime format. Finally, I utilized the time delta module to change my time into Eastern Daylight Time. 
In order to process the comment data, I had to clean the characters used in the comments. The comments were littered with punctuation, extra spaces, emojis, and links. I parsed through the dataset to remove all the unwanted characters and stored the cleaned in a new column in my dataframe. To further clean the dataset, I removed the English stop words and split the long string into a list of words.
The financial obtained from Yahoo Finance was clean and organized when I accessed the data through their API. However, in order to merge dataframes I had to change the index of the dataset (the dates). The dates were in Timestamp format whereas the rest of the dates were kept in string format. In order to make the dataset compatible for merging, I reset the index of the dataframe and converted the dates from the Timestamp format to string format using type casting, string splicing, and list comprehension.

# Use Cases

The ways for the user to interact with this code include 
1.	Scape reddit data (a user can specify the subreddit name and the number of posts they would like)
2.	Clean all the comment data by stripping punctuation, extra spaces, emojis, and links given a string
3.	Create a word clouds for any given month
4.	Retreive stock data given a company/ETF, start date, and end date
5.	Plot the number of mentions of company on a subreddit versus its share price in the given timeframe
6.	Plot the number of mentions of any give word vs the SPY ETF or ^VIX ETF.
7.	Render a graph listing the Top 10 words used in the posts for a specific day (given the day)
8.	Plot the daily Open, High, Low, and Close for the financial data of any given company/ETF


# Conclusions

One of the goals of this was to use natural language processing in order to analyze the most frequent words. The word clouds were an excellent visualization technique that summarized the vast amount of data on r/wallstreetbets. I originally hypothesized that the Calls-to-Puts would increase over time. The results of my findings did show that this was in fact true and I went on to find that there was in fact a positive, however weak correlation between the Calls-to-Puts ratio and the closing price of the SPY ETF. Moreover, this trend was affirmed by the negative correlation between the price of the SPY ETF and the number of occurrences of the word “Puts” in the subreddit. The motivation of this analysis was to better understand the social media response to financial gains and losses. I was surprised to see several general trends on social media that do correlate with the stock market. 
	However, I did see the results of some of analysis failed to provide any significant results. Some patterns I thought I would find between the change in the share price and the company’s popularity failed to produce any significant patterns. The art of finding patterns and the predictability of stock price movement proved to be far more complex than I originally thought, and this was in fact reflected in other part of my data analysis. When analyzing the frequency of key words in trading that were seen in the word clouds did not show any patterns that were in relation to the relative changes of the market. 
 	This analysis is subjected several limitations. Using one subreddit to encompass the opinions of the population of investors produces skewed results as this is not a sample that is representative of the entire public. In addition, as I was looking through only the top posts for this analysis, there could be other posts that reflect different sentiment that I never analyzed. Moreover, there are so many variables that influence the direction of the stock market and I only analyzed one aspect of how the stock market moves. There are a vast number of confounding variables that I have not considered in the scope of the project. In addition, using more powerful natural processing techniques such as n-grams to better understand the context in which these words were being used in could provide better results. One of the reasons I was drawn to this analysis is because 2020 saw some of the largest changes with respect to the stock market. However, these results may only reflect what happened in 2020 as opposed to a general trend for every year. 

# References


1.	Gan, C. J. (2020, October 19). 5 Quant Strategies used by a Wall Street Trader. Retrieved from https://medium.com/datadriveninvestor/5-strategies-in-quant-trading-algorithms-f4f782d152e2

2.	Samrega. (2020, October 07). How Robinhood and Covid opened the floodgates for 13 million amateur stock traders. Retrieved from https://www.cnbc.com/2020/10/07/how-robinhood-and-covid-introduced-millions-to-the-stock-market.html 

3.	The Python Reddit API Wrapper¶. (n.d.). Retrieved from https://praw.readthedocs.io/en/latest/
