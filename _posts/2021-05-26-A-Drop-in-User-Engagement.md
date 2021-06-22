---
layout: post
title: Investigating A Drop In User Engagement
author: Jonathan Logan Moran
categories: portfolio
tags: plotly python SQL ipynb jekyll
permalink: /a-drop-in-user-engagement
description: "Part 1 in a three-part study exploring the world of Yammer– a workplace communication platform. In this study, we investigate a drop in user engagement on the platform by using product analytics and data science tools."
---

# Investigating a Drop in User Engagement

In this study, we will be using a few common technologies:
*   SQL (SQLite) for data manipulation/exploration
*   Python (pandas) for data frames
*   Plotly/Jupyter notebook for interactive charts/data visualization

## Programming

```python
# prerequisites
!pip install pandasql
!pip install plotly
```

```python
# data tools
import pandas as pd
import pandasql					# for querying pandas dataframes
from pandasql import sqldf

# plotting tools
import plotly.graph_objects as go
import plotly.express as px
```

```python
import plotly.io as pio				# output Plotly charts to HMTL
```

## Task overview
This notebook features the first case study in a series of three that dives into the Yammer dataset. For our analysis we will be considering Yammer's core metrics (user engagement, retention and growth) and analyzing product-specific usage metrics (e.g. number of times someone views another user's profile). For clarification, Yammer is a social network for the workplace whose primary goal is to drive better product and business decisions using data. While this specific dataset is fabricated due to privacy and security reasons, it is similar in structure to Yammer's actual data. With all that said–let's begin our study...

## Collecting our dataset


```python
emails_path = '../src/yammer_emails.csv'
events_path = '../src/yammer_events.csv'
users_path = '../src/yammer_users.csv'
```


```python
yammer_emails = pd.read_csv(emails_path, sep=',', header=0)
yammer_emails
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>occurred_at</th>
      <th>action</th>
      <th>user_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2014-05-06 09:30:00</td>
      <td>sent_weekly_digest</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>2014-05-13 09:30:00</td>
      <td>sent_weekly_digest</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2014-05-20 09:30:00</td>
      <td>sent_weekly_digest</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>2014-05-27 09:30:00</td>
      <td>sent_weekly_digest</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>2014-06-03 09:30:00</td>
      <td>sent_weekly_digest</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>90384</th>
      <td>18814.0</td>
      <td>2014-08-31 12:12:26</td>
      <td>email_open</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>90385</th>
      <td>18814.0</td>
      <td>2014-08-31 12:12:57</td>
      <td>email_clickthrough</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>90386</th>
      <td>18815.0</td>
      <td>2014-08-31 13:39:56</td>
      <td>sent_reengagement_email</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>90387</th>
      <td>18815.0</td>
      <td>2014-08-31 13:40:14</td>
      <td>email_open</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>90388</th>
      <td>18815.0</td>
      <td>2014-08-31 13:40:47</td>
      <td>email_clickthrough</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>90389 rows × 4 columns</p>
</div>




```python
yammer_events = pd.read_csv(events_path, sep=',', header=0)
yammer_events
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>occurred_at</th>
      <th>event_type</th>
      <th>event_name</th>
      <th>location</th>
      <th>device</th>
      <th>user_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10522.0</td>
      <td>2014-05-02 11:02:39</td>
      <td>engagement</td>
      <td>login</td>
      <td>Japan</td>
      <td>dell inspiron notebook</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10522.0</td>
      <td>2014-05-02 11:02:53</td>
      <td>engagement</td>
      <td>home_page</td>
      <td>Japan</td>
      <td>dell inspiron notebook</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10522.0</td>
      <td>2014-05-02 11:03:28</td>
      <td>engagement</td>
      <td>like_message</td>
      <td>Japan</td>
      <td>dell inspiron notebook</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10522.0</td>
      <td>2014-05-02 11:04:09</td>
      <td>engagement</td>
      <td>view_inbox</td>
      <td>Japan</td>
      <td>dell inspiron notebook</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10522.0</td>
      <td>2014-05-02 11:03:16</td>
      <td>engagement</td>
      <td>search_run</td>
      <td>Japan</td>
      <td>dell inspiron notebook</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>340827</th>
      <td>18815.0</td>
      <td>2014-08-31 13:41:46</td>
      <td>engagement</td>
      <td>like_message</td>
      <td>Ireland</td>
      <td>dell inspiron notebook</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>340828</th>
      <td>18815.0</td>
      <td>2014-08-31 13:42:11</td>
      <td>engagement</td>
      <td>home_page</td>
      <td>Ireland</td>
      <td>dell inspiron notebook</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>340829</th>
      <td>18815.0</td>
      <td>2014-08-31 13:42:43</td>
      <td>engagement</td>
      <td>send_message</td>
      <td>Ireland</td>
      <td>dell inspiron notebook</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>340830</th>
      <td>18815.0</td>
      <td>2014-08-31 13:43:07</td>
      <td>engagement</td>
      <td>home_page</td>
      <td>Ireland</td>
      <td>dell inspiron notebook</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>340831</th>
      <td>18815.0</td>
      <td>2014-08-31 13:43:42</td>
      <td>engagement</td>
      <td>like_message</td>
      <td>Ireland</td>
      <td>dell inspiron notebook</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>340832 rows × 7 columns</p>
</div>




```python
yammer_users = pd.read_csv(users_path, sep=',', header=0)
yammer_users
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>created_at</th>
      <th>company_id</th>
      <th>language</th>
      <th>activated_at</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2013-01-01 20:59:39</td>
      <td>5737.0</td>
      <td>english</td>
      <td>2013-01-01 21:01:07</td>
      <td>active</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2013-01-01 13:07:46</td>
      <td>28.0</td>
      <td>english</td>
      <td>NaN</td>
      <td>pending</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2013-01-01 10:59:05</td>
      <td>51.0</td>
      <td>english</td>
      <td>NaN</td>
      <td>pending</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2013-01-01 18:40:36</td>
      <td>2800.0</td>
      <td>german</td>
      <td>2013-01-01 18:42:02</td>
      <td>active</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>2013-01-01 14:37:51</td>
      <td>5110.0</td>
      <td>indian</td>
      <td>2013-01-01 14:39:05</td>
      <td>active</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19061</th>
      <td>19061.0</td>
      <td>2014-08-31 13:21:16</td>
      <td>2156.0</td>
      <td>chinese</td>
      <td>2014-08-31 13:22:50</td>
      <td>active</td>
    </tr>
    <tr>
      <th>19062</th>
      <td>19062.0</td>
      <td>2014-08-31 19:21:23</td>
      <td>7520.0</td>
      <td>spanish</td>
      <td>NaN</td>
      <td>pending</td>
    </tr>
    <tr>
      <th>19063</th>
      <td>19063.0</td>
      <td>2014-08-31 07:10:41</td>
      <td>72.0</td>
      <td>spanish</td>
      <td>2014-08-31 07:12:09</td>
      <td>active</td>
    </tr>
    <tr>
      <th>19064</th>
      <td>19064.0</td>
      <td>2014-08-31 17:45:18</td>
      <td>2.0</td>
      <td>english</td>
      <td>NaN</td>
      <td>pending</td>
    </tr>
    <tr>
      <th>19065</th>
      <td>19065.0</td>
      <td>2014-08-31 19:29:19</td>
      <td>8352.0</td>
      <td>italian</td>
      <td>NaN</td>
      <td>pending</td>
    </tr>
  </tbody>
</table>
<p>19066 rows × 6 columns</p>
</div>



## The problem
It’s Tuesday, September 2, 2014. You’ve just been informed by the head of the Yammer Product team that user engagement is down. Here's the chart of weekly active users that you've been provided:


```python
q = """
SELECT strftime('%Y-%m-%d', occurred_at, 'weekday 0', '-6 days') AS week, COUNT(DISTINCT user_id) AS weekly_active_users
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-09-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1;
"""
```


```python
weekly_active_users = pandasql.sqldf(q, globals())
```


```python
week = weekly_active_users['week']
count_users = weekly_active_users['weekly_active_users']
```


```python
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=week, y=count_users, mode='lines', name='num_users'))
fig1.update_layout(title='Weekly Active Users', xaxis_title='Week of', yaxis_title='Number of users')
pio.write_html(fig1, file='figures/fig1.html', auto_open=True)
```

{% include 2021-05-26-A-Drop-in-User-Engagement-fig1.html %}

The above chart shows the number of engaged users each week. Yammer defines engagement as having made some type of server call by interacting with the product (showed in the data as events of type `engagement`). Any point in this chart can be interpreted as "the number of users who logged at least one engagement event during the week starting on that date".

## Analysis
We are responsible for determining what caused the dip at the end of the chart shown above, and, if appropriate, recommending solutions for the problem.

In this study, we will be considering the following metrics:
*   Engagement by Factors of Time
*   New Users (Activation/Growth Rate)
*   Engagement by Event Type
*   Engagement by Region
*   Engagement by Product/Device Type
*   Email Open and Click-Through Rate

### _1. Engagement by Factors of Time_
"Do we see the engagement levels gradually decreasing over time or only a one time sudden decrease? Is the decrease specific to certain days of the week or certain times of the day?"

### Daily Engagement Rate


```python
# engagement rate = num of engagments / num of users
q = """
SELECT strftime('%Y-%m-%d', occurred_at) AS day, COUNT(*) * 1.00 / COUNT(DISTINCT user_id) AS engagement_rate
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-09-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1
"""
```


```python
daily_engagement = pandasql.sqldf(q, globals())
```


```python
daily_engagement.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>engagement_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-05-01</td>
      <td>9.808874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-02</td>
      <td>10.765363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-03</td>
      <td>8.903448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-04</td>
      <td>8.734177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-05</td>
      <td>10.050584</td>
    </tr>
  </tbody>
</table>
</div>




```python
day = daily_engagement['day']
engagement_rate = daily_engagement['engagement_rate']
```


```python
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=day, y=engagement_rate, mode='lines', name='engagement_rate'))
fig2.update_layout(title='Daily Engagement', xaxis_title='Day', yaxis_title='Engagement rate')
pio.write_html(fig2, file='figures/fig2.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig2.html %}

### Engagement by Week


```python
q = """
SELECT strftime('%w', occurred_at) AS dow,
       CASE strftime('%w', occurred_at)
       WHEN '0' THEN 'Sun'
       WHEN '1' THEN 'Mon'
       WHEN '2' THEN 'Tues'
       WHEN '3' THEN 'Wed'
       WHEN '4' THEN 'Thurs'
       WHEN '5' THEN 'Fri'
       WHEN '6' THEN 'Sat'
       ELSE NULL END AS day_of_week,
       COUNT(*) * 1.00 / COUNT(DISTINCT user_id) AS active_users
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-09-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1
"""
```


```python
dow_engagement = pandasql.sqldf(q, globals())
```


```python
dow_engagement.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dow</th>
      <th>day_of_week</th>
      <th>active_users</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sun</td>
      <td>9.633461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Mon</td>
      <td>14.766006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Tues</td>
      <td>16.405758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Wed</td>
      <td>17.235331</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Thurs</td>
      <td>17.961189</td>
    </tr>
  </tbody>
</table>
</div>




```python
day = dow_engagement['day_of_week']
engagement_rate = dow_engagement['active_users']
```


```python
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=day, y=engagement_rate))
fig3.update_layout(title='Engagement by Day of Week', xaxis_title='Day of Week', yaxis_title='Engagement rate')
pio.write_html(fig3, file='figures/fig3.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig3.html %}

### Engagement by Time of Day


```python
q = """
SELECT strftime('%H', occurred_at) AS hour,
       COUNT(*) * 1.00 / COUNT(DISTINCT user_id) AS engagement_rate
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-9-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1
"""
```


```python
time_day = pandasql.sqldf(q, globals())
```


```python
time_day.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>engagement_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00</td>
      <td>9.787736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>9.410628</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02</td>
      <td>8.578125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03</td>
      <td>9.088710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04</td>
      <td>9.169014</td>
    </tr>
  </tbody>
</table>
</div>




```python
hour = time_day['hour']
engagement_rate = time_day['engagement_rate']
```


```python
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=hour, y=engagement_rate))
fig4.update_layout(title='Engagement by Time of Day', xaxis_title='Hour', yaxis_title='Engagement rate')
pio.write_html(fig4, file='figures/fig4.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig4.html %}

Conclusions:

*   From the above graphs, we can see that the drop in engagement was gradual over the span of a few months and not localized to a particular day or time. We can assume that this is not likely due to a one-time server/technical outage.
*   The Time of Day chart seems consistent in engagement aside from one outlier–9 AM, when users appear to have the most acitivty at this time.
*   Engagement by Day of Week tells us that our users have the least activty on the days of Saturday and Sunday, which make sense since Yammer is typically used during work hours.

### _2. New Users (Activation and Growth Rate)_
"Since the drop in users were gradual over time and not localized to a particular day of the week or time of day, we should now move on and identify if there is an issue with acquiring new users or activating new users"


```python
q = """
SELECT
  strftime('%Y-%m-%d', created_at) AS day,
  COUNT(*) AS all_users,
  COUNT(CASE WHEN activated_at IS NOT NULL THEN user_id ELSE NULL END) AS activated_users
FROM yammer_users
WHERE created_at >= '2014-05-01 00:00:00'
AND created_at < '2014-09-01 00:00:00'
GROUP BY 1
ORDER BY 1
"""
```


```python
growth = pandasql.sqldf(q, globals())
```


```python
day = growth['day']
all_users = growth['all_users']
activated_users = growth['activated_users']
```


```python
growth.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>all_users</th>
      <th>activated_users</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-05-01</td>
      <td>73</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-02</td>
      <td>57</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-03</td>
      <td>19</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-04</td>
      <td>22</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-05</td>
      <td>58</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=day, y=all_users, mode='lines', name='all_users'))
fig5.add_trace(go.Scatter(x=day, y=activated_users, mode='lines', name='activated_users'))
fig5.update_layout(title='Daily signups', xaxis_title='Date', yaxis_title='Number of users')
pio.write_html(fig5, file='figures/fig5.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig5.html %}

Conclusions:
*   The chart above indicates that the growth rate appears normal. We can assume that there are no sign up issues.
*   We will look into the drop in engagement from our existing users, as Yammer's goal metric is satisfied (Yammer is providing the intended value for new customers).

### _3. Engagement by Event Type_
By looking at the assumed UX flow, there are four major types of engagement events:
* signing up as a new user
* logging in to homepage as an existing user
* inbox and message related engagement
* search and click related engagement

We want to group these together, and see if there is a more pronounced drop in engagement for either groups


```python
q = """
SELECT DISTINCT event_name
FROM yammer_events
"""
```


```python
events = pandasql.sqldf(q, globals())
```


```python
events.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>login</td>
    </tr>
    <tr>
      <th>1</th>
      <td>home_page</td>
    </tr>
    <tr>
      <th>2</th>
      <td>like_message</td>
    </tr>
    <tr>
      <th>3</th>
      <td>view_inbox</td>
    </tr>
    <tr>
      <th>4</th>
      <td>search_run</td>
    </tr>
    <tr>
      <th>5</th>
      <td>send_message</td>
    </tr>
    <tr>
      <th>6</th>
      <td>search_autocomplete</td>
    </tr>
    <tr>
      <th>7</th>
      <td>search_click_result_10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>create_user</td>
    </tr>
    <tr>
      <th>9</th>
      <td>enter_email</td>
    </tr>
    <tr>
      <th>10</th>
      <td>enter_info</td>
    </tr>
    <tr>
      <th>11</th>
      <td>complete_signup</td>
    </tr>
    <tr>
      <th>12</th>
      <td>search_click_result_7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>search_click_result_8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>search_click_result_1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>search_click_result_3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>search_click_result_2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>search_click_result_5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>search_click_result_6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>search_click_result_9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>search_click_result_4</td>
    </tr>
  </tbody>
</table>
</div>




```python
q = """
SELECT
  strftime('%Y-%m-%d', occurred_at, 'weekday 0', '-6 days') AS week_of,
  COUNT(DISTINCT CASE WHEN event_name IN ('create_user', 'enter_email', 'enter_info', 'complete_signup') THEN user_id ELSE NULL END) AS signup,
  COUNT(DISTINCT CASE WHEN event_name IN ('login', 'home_page') THEN user_id ELSE NULL END) AS login,
  COUNT(DISTINCT CASE WHEN event_name IN ('view_inbox', 'like_message', 'send_message') THEN user_id ELSE NULL END) AS inbox,
  COUNT(DISTINCT CASE WHEN event_name IN ('search_run', 'search_autocomplete', 'search_click_result_1', 'search_click_result_2', 'search_click_result_3', 'search_click_result_4', 'search_click_result_5', 'search_click_result_6', 'search_click_result_7', 'search_click_result_8', 'search_click_result_9', 'search_click_result_10') THEN user_id ELSE NULL END) AS search_and_click
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-09-01 00:00:00'
GROUP BY 1
ORDER BY 1
"""
```


```python
event_engagement = pandasql.sqldf(q, globals())
```


```python
week_of = event_engagement['week_of']
signup = event_engagement['signup']
login = event_engagement['login']
inbox = event_engagement['inbox']
search_and_click = event_engagement['search_and_click']
```


```python
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=week_of, y=signup, mode='lines', name='signup'))
fig6.add_trace(go.Scatter(x=week_of, y=login, mode='lines', name='login'))
fig6.add_trace(go.Scatter(x=week_of, y=inbox, mode='lines', name='inbox'))
fig6.add_trace(go.Scatter(x=week_of, y=search_and_click, mode='lines', name='search and click'))
fig6.update_layout(title='Engagement by Event Type', xaxis_title='Week of', yaxis_title='Number of users')
pio.write_html(fig6, file='figures/fig6.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig6.html %}

Conclusions:
*   We observe a decrease in engagement across all events, namely _homepage_ and _message_ activity.
*   Users appear to be navigating through the signup funnel, as we see no decrease in activty in this category.

### _4. Engagement by Region_
"If the drop is localized to a particular region then there could be an issue related to that."


```python
q = """
-- Selects Top 5 countries of all-time
WITH region AS (SELECT location, COUNT(user_id) AS num_users FROM yammer_events WHERE occurred_at >= '2014-05-01 00:00:00' AND occurred_at < '2014-09-01 00:00:00' AND event_type = 'engagement' GROUP BY 1 ORDER BY 2 DESC),
top AS (SELECT region.location, region.num_users, ROW_NUMBER() OVER (ORDER BY num_users DESC) AS rank FROM region)
SELECT strftime('%Y-%m-%d', e.occurred_at, 'weekday 0', '-6 days') AS week_of,
       e.location,
       COUNT(DISTINCT e.user_id) AS num_users
FROM yammer_events e
INNER JOIN top
ON e.location = top.location
WHERE occurred_at >= '2014-05-01 00:00:00' 
AND occurred_at < '2014-09-01 00:00:00' 
AND event_type = 'engagement'
AND top.rank <=5
GROUP BY 1, 2
ORDER BY 1, 3 DESC
"""
```


```python
top_regions = pandasql.sqldf(q, globals())
```


```python
top_regions.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week_of</th>
      <th>location</th>
      <th>num_users</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-28</td>
      <td>United States</td>
      <td>203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-28</td>
      <td>Japan</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-04-28</td>
      <td>Germany</td>
      <td>42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-04-28</td>
      <td>France</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-04-28</td>
      <td>United Kingdom</td>
      <td>33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-05</td>
      <td>United States</td>
      <td>294</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014-05-05</td>
      <td>Japan</td>
      <td>76</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2014-05-05</td>
      <td>Germany</td>
      <td>72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2014-05-05</td>
      <td>France</td>
      <td>53</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014-05-05</td>
      <td>United Kingdom</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig7 = px.line(top_regions, x="week_of", y="num_users", color="location", title='Weekly Engagement by Region')
#iplot(fig7)
pio.write_html(fig7, file='figures/fig7.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig7.html %}

Conclusions:
*   We considered engagement across our top five regions. From the graph, we determine that the United States and the most signficant drop in activtiy.
*   Because of this observation, we will refer to the Engineering and Marketing teams to determine if a product release or campagin is responsible for the decrease in active U.S. users.

### _5. Engagement by Device/Product Type_
"It is also good to check if the drop in engagement rate is localized to a type of device. We'd want to compare laptops vs phones vs tablets and then perhaps iOS vs androids, or macs vs PCs."

### Engagement by Device Type
We'll categorize our users by their device type, specifically:
* Desktop
* Phone
* Tablet


```python
q = """
SELECT DISTINCT device
FROM yammer_events
"""
```


```python
devices = pandasql.sqldf(q, globals())
```


```python
devices.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dell inspiron notebook</td>
    </tr>
    <tr>
      <th>1</th>
      <td>iphone 5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>iphone 4s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>windows surface</td>
    </tr>
    <tr>
      <th>4</th>
      <td>macbook air</td>
    </tr>
    <tr>
      <th>5</th>
      <td>iphone 5s</td>
    </tr>
    <tr>
      <th>6</th>
      <td>macbook pro</td>
    </tr>
    <tr>
      <th>7</th>
      <td>kindle fire</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ipad mini</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nexus 7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>nexus 5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>samsung galaxy s4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lenovo thinkpad</td>
    </tr>
    <tr>
      <th>13</th>
      <td>samsumg galaxy tablet</td>
    </tr>
    <tr>
      <th>14</th>
      <td>acer aspire notebook</td>
    </tr>
    <tr>
      <th>15</th>
      <td>asus chromebook</td>
    </tr>
    <tr>
      <th>16</th>
      <td>htc one</td>
    </tr>
    <tr>
      <th>17</th>
      <td>nokia lumia 635</td>
    </tr>
    <tr>
      <th>18</th>
      <td>samsung galaxy note</td>
    </tr>
    <tr>
      <th>19</th>
      <td>acer aspire desktop</td>
    </tr>
    <tr>
      <th>20</th>
      <td>mac mini</td>
    </tr>
    <tr>
      <th>21</th>
      <td>hp pavilion desktop</td>
    </tr>
    <tr>
      <th>22</th>
      <td>dell inspiron desktop</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ipad air</td>
    </tr>
    <tr>
      <th>24</th>
      <td>amazon fire phone</td>
    </tr>
    <tr>
      <th>25</th>
      <td>nexus 10</td>
    </tr>
  </tbody>
</table>
</div>




```python
q = """
SELECT 
  strftime('%Y-%m-%d', occurred_at, 'weekday 0', '-6 days') AS week_of,
  COUNT(DISTINCT user_id) AS weekly_active_users,
  COUNT(DISTINCT CASE WHEN device IN ('dell inspiron notebook', 'macbook air', 'macbook pro', 'lenovo thinkpad', 'acer aspire notebook', 'asus chromebook', 'acer aspire desktop', 'mac mini', 'hp pavillion desktop', 'dell inspiron desktop') THEN user_id ELSE NULL END) AS desktop,
  COUNT(DISTINCT CASE WHEN device IN ('iphone 5', 'iphone 4s', 'iphone 5s', 'nexus 5', 'samsung galaxy s4', 'htc one', 'nokia lumia 635', 'samsung galaxy note', 'amazon fire phone') THEN user_id ELSE NULL END) AS phone,
  COUNT(DISTINCT CASE WHEN device IN ('windows surface', 'kindle fire', 'ipad mini', 'nexus 7', 'samsung galaxy tablet', 'ipad air', 'nexus 10') THEN user_id ELSE NULL END) AS tablet
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00'
AND occurred_at < '2014-09-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1
"""
```


```python
device_engagement = pandasql.sqldf(q, globals())
```


```python
week_of = device_engagement['week_of']
weekly_active_users = device_engagement['weekly_active_users']
desktop = device_engagement['desktop']
phone = device_engagement['phone']
tablet = device_engagement['tablet']
```


```python
fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=week_of, y=weekly_active_users, mode='lines', name='weekly active users'))
fig8.add_trace(go.Scatter(x=week_of, y=desktop, mode='lines', name='desktop'))
fig8.add_trace(go.Scatter(x=week_of, y=phone, mode='lines', name='phone'))
fig8.add_trace(go.Scatter(x=week_of, y=tablet, mode='lines', name='tablet'))
fig8.update_layout(title='Engagement by Device Type', xaxis_title='Week of', yaxis_title='Number of users')
pio.write_html(fig8, file='figures/fig8.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig8.html %}

### Engagement by Product Type
From the above plot, we can see that our engagement decreased most significantly on the mobile ("phone") Yammer platform.

We'll further investigate by looking at the differences in engagement across two mobile OS types:
* iOS
* Other (Android/Windows)


```python
q = """
SELECT
  strftime('%Y-%m-%d', occurred_at, 'weekday 0', '-6 days') AS week_of,
  COUNT(DISTINCT CASE WHEN device IN ('iphone 4s', 'iphone 5', 'iphone 5s') THEN user_id ELSE NULL END) AS "iOS",
  COUNT(DISTINCT CASE WHEN device IN ('nexus 5', 'samsung galaxy s4', 'htc one', 'nokia lumia 635', 'samsung galaxy note', 'amazon fire phone') THEN user_id ELSE NULL END) AS "Other"
FROM yammer_events
WHERE occurred_at >= '2014-05-01 00:00:00' 
AND occurred_at < '2014-09-01 00:00:00'
AND event_type = 'engagement'
GROUP BY 1
ORDER BY 1
"""
```


```python
os_engagement = pandasql.sqldf(q, globals())
```


```python
week_of = os_engagement['week_of']
iOS = os_engagement['iOS']
Other = os_engagement['Other']
```


```python
fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=week_of, y=iOS, mode='lines', name='iOS'))
fig9.add_trace(go.Scatter(x=week_of, y=Other, mode='lines', name='Other'))
fig9.update_layout(title='Engagement by Phone OS', xaxis_title='Week of', yaxis_title='Number of users')
pio.write_html(fig9, file='figures/fig9.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig9.html %}

Conclusions:
*   Engagement appears to be down across all device types, but we observe a more significant drop in `phone` users.
*   Again, we need to check with the Engineering team to see if any recent product releases on the mobile platform are responsible for this decline in engagement.

### _5. Cohort Analysis_
"Since activation rate is normal, we know the issue is not associated with growth (acquiring new users). The drop will likely be due to disengagement from existing users, so we should do a cohort analysis allowing us to compare users at different life stages with Yammer."


```python
q = """
SELECT 
  strftime('%Y-%m-%d', ue.occurred_at, 'weekday 0', '-6 days') AS week_of,
  COUNT(DISTINCT CASE WHEN ue.user_age < 7 THEN ue.user_id ELSE NULL END) AS '<1 week old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 7 AND ue.user_age < 14 THEN ue.user_id ELSE NULL END) AS '1 week old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 14 AND ue.user_age < 21 THEN ue.user_id ELSE NULL END) AS '2 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 21 AND ue.user_age < 28 THEN ue.user_id ELSE NULL END) AS '3 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 28 AND ue.user_age < 35 THEN ue.user_id ELSE NULL END) AS '4 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 35 AND ue.user_age < 42 THEN ue.user_id ELSE NULL END) AS '5 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 42 AND ue.user_age < 49 THEN ue.user_id ELSE NULL END) AS '6 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 49 AND ue.user_age < 56 THEN ue.user_id ELSE NULL END) AS '7 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 56 AND ue.user_age < 63 THEN ue.user_id ELSE NULL END) AS '8 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 63 AND ue.user_age < 70 THEN ue.user_id ELSE NULL END) AS '9 weeks old',
  COUNT(DISTINCT CASE WHEN ue.user_age >= 70 THEN ue.user_id ELSE NULL END) AS '10+ weeks old'
FROM (SELECT e.occurred_at, u.user_id, CAST((julianday('2014-09-01') - julianday(u.activated_at)) AS INT) AS user_age
      FROM yammer_users u
      JOIN yammer_events e
      ON e.user_id = u.user_id
      WHERE e.occurred_at >= '2014-05-01 00:00:00'
      AND e.occurred_at < '2014-09-01 00:00:00'
      AND e.event_type = 'engagement'
      AND u.activated_at IS NOT NULL) ue
GROUP BY 1
ORDER BY 1
"""
```


```python
age_cohort = pandasql.sqldf(q, globals())
```


```python
age_cohort.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week_of</th>
      <th>&lt;1 week old</th>
      <th>1 week old</th>
      <th>2 weeks old</th>
      <th>3 weeks old</th>
      <th>4 weeks old</th>
      <th>5 weeks old</th>
      <th>6 weeks old</th>
      <th>7 weeks old</th>
      <th>8 weeks old</th>
      <th>9 weeks old</th>
      <th>10+ weeks old</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>701</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1054</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-06-02</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1173</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014-06-09</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1219</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2014-06-16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1263</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2014-06-23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>210</td>
      <td>1039</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014-06-30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>199</td>
      <td>151</td>
      <td>921</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2014-07-07</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>223</td>
      <td>130</td>
      <td>100</td>
      <td>902</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2014-07-14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215</td>
      <td>152</td>
      <td>82</td>
      <td>62</td>
      <td>834</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2014-07-21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>228</td>
      <td>144</td>
      <td>95</td>
      <td>60</td>
      <td>44</td>
      <td>792</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014-07-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>234</td>
      <td>156</td>
      <td>91</td>
      <td>83</td>
      <td>43</td>
      <td>30</td>
      <td>806</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014-08-04</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189</td>
      <td>154</td>
      <td>82</td>
      <td>52</td>
      <td>52</td>
      <td>34</td>
      <td>24</td>
      <td>679</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2014-08-11</td>
      <td>0</td>
      <td>0</td>
      <td>250</td>
      <td>126</td>
      <td>94</td>
      <td>59</td>
      <td>33</td>
      <td>39</td>
      <td>33</td>
      <td>19</td>
      <td>562</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014-08-18</td>
      <td>0</td>
      <td>259</td>
      <td>163</td>
      <td>69</td>
      <td>64</td>
      <td>40</td>
      <td>19</td>
      <td>26</td>
      <td>26</td>
      <td>15</td>
      <td>522</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2014-08-25</td>
      <td>266</td>
      <td>173</td>
      <td>82</td>
      <td>48</td>
      <td>47</td>
      <td>31</td>
      <td>20</td>
      <td>23</td>
      <td>14</td>
      <td>15</td>
      <td>475</td>
    </tr>
  </tbody>
</table>
</div>




```python
week_of = age_cohort['week_of']
y0 = age_cohort['<1 week old']
y1 = age_cohort['1 week old']
y2 = age_cohort['2 weeks old']
y3 = age_cohort['3 weeks old']
y4 = age_cohort['4 weeks old']
y5 = age_cohort['5 weeks old']
y6 = age_cohort['6 weeks old']
y7 = age_cohort['7 weeks old']
y8 = age_cohort['8 weeks old']
y9 = age_cohort['9 weeks old']
y10 = age_cohort['10+ weeks old']
```


```python
fig10 = go.Figure()
fig10.add_trace(go.Scatter(x=week_of, y=y0, mode='lines', name='<1 week old'))
fig10.add_trace(go.Scatter(x=week_of, y=y1, mode='lines', name='1 week old'))
fig10.add_trace(go.Scatter(x=week_of, y=y2, mode='lines', name='2 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y3, mode='lines', name='3 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y4, mode='lines', name='4 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y5, mode='lines', name='5 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y6, mode='lines', name='6 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y7, mode='lines', name='7 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y8, mode='lines', name='8 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y9, mode='lines', name='9 weeks old'))
fig10.add_trace(go.Scatter(x=week_of, y=y10, mode='lines', name='10+ weeks old'))
fig10.update_layout(title='Engagement by User Age Cohort', xaxis_title='Week of', yaxis_title='Number of Users')
pio.write_html(fig10, file='figures/fig10.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig10.html %}

Conclusions:
*   From this Cohort Analysis, we observe a decrease in engagement specific to our existing users in the 10+ weeks old account age group.
*   Looking at the engagement rate of new users, we don't have any data to suggest that feature awareness is an issue.

### _6. Email Open and CTR_
Yammer sends users two types of email to increase engagement:
* Weekly digest emails (`sent_weekly_digest`)
* Re-engagement emails (`sent_reengagement_email`)

We want to see if something has gone wrong with these emails by looking into their open and click-through rates.


```python
q = """
SELECT DISTINCT action
FROM yammer_emails
"""
```


```python
emails = pandasql.sqldf(q, globals())
```


```python
emails.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sent_weekly_digest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>email_open</td>
    </tr>
    <tr>
      <th>2</th>
      <td>email_clickthrough</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sent_reengagement_email</td>
    </tr>
  </tbody>
</table>
</div>



Note:
Some preliminary analysis done on `email_open` rates suggest that *>97%* of all engagement emails sent are opened _within the first hour_. Furthermore, less than 2% of engagment emails are opened after the 24-hour mark. Because of this, we'll be looking at `email_open` and `email_clickthrough` rates that occur within a 24-hour window.


```python
q = """
SELECT 
  strftime('%Y-%m-%d', e.occurred_at, 'weekday 0', '-6 days') AS week_of,
  COUNT(CASE WHEN e.action = 'sent_weekly_digest' THEN e.user_id ELSE NULL END) AS emails_sent,
  COUNT(CASE WHEN e.action = 'sent_weekly_digest' THEN e1.user_id ELSE NULL END) AS emails_opened,
  COUNT(CASE WHEN e.action = 'sent_weekly_digest' THEN e2.user_id ELSE NULL END) AS emails_ct,
  COUNT(CASE WHEN e.action = 'sent_reengagement_email' THEN e.user_id ELSE NULL END) AS reengagements_sent,
  COUNT(CASE WHEN e.action = 'sent_reengagement_email' THEN e1.user_id ELSE NULL END) AS reengagements_opened,
  COUNT(CASE WHEN e.action = 'sent_reengagement_email' THEN e2.user_id ELSE NULL END) AS reengagements_ct
FROM yammer_emails e
LEFT JOIN yammer_emails e1
  ON e.user_id = e1.user_id
    AND e1.action = 'email_open'
    AND e1.occurred_at >= e.occurred_at
    AND e1.occurred_at < datetime(e.occurred_at, '+24 hours')
LEFT JOIN yammer_emails e2
  ON e.user_id = e2.user_id
    AND e2.action = 'email_clickthrough'
    AND e2.occurred_at >= e.occurred_at
    AND e2.occurred_at < datetime(e.occurred_at, '+24 hours')
WHERE e.occurred_at >= '2014-05-01 00:00:00'
AND e.occurred_at < '2014-09-01 00:00:00'
GROUP BY 1
ORDER BY 1
"""
```


```python
email_counts = pandasql.sqldf(q, globals())
```


```python
email_counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week_of</th>
      <th>emails_sent</th>
      <th>emails_opened</th>
      <th>emails_ct</th>
      <th>reengagements_sent</th>
      <th>reengagements_opened</th>
      <th>reengagements_ct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-28</td>
      <td>908</td>
      <td>246</td>
      <td>105</td>
      <td>98</td>
      <td>86</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-05</td>
      <td>2602</td>
      <td>776</td>
      <td>306</td>
      <td>164</td>
      <td>143</td>
      <td>128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-12</td>
      <td>2665</td>
      <td>809</td>
      <td>332</td>
      <td>175</td>
      <td>162</td>
      <td>147</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-19</td>
      <td>2733</td>
      <td>834</td>
      <td>348</td>
      <td>179</td>
      <td>161</td>
      <td>150</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-26</td>
      <td>2822</td>
      <td>868</td>
      <td>314</td>
      <td>179</td>
      <td>158</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
</div>




```python
week_of = email_counts['week_of']
emails_sent = email_counts['emails_sent']
emails_opened = email_counts['emails_opened']
emails_ct = email_counts['emails_ct']
reengagements_sent = email_counts['reengagements_sent']
reengagements_opened = email_counts['reengagements_opened']
reengagements_ct = email_counts['reengagements_ct']
```


```python
fig11 = go.Figure()
fig11.add_trace(go.Scatter(x=week_of, y=emails_sent, mode='lines', name='emails sent'))
fig11.add_trace(go.Scatter(x=week_of, y=emails_opened, mode='lines', name='emails opened'))
fig11.add_trace(go.Scatter(x=week_of, y=emails_ct, mode='lines', name='emails clicked-through'))
fig11.add_trace(go.Scatter(x=week_of, y=reengagements_sent, mode='lines', name='reengagements sent'))
fig11.add_trace(go.Scatter(x=week_of, y=reengagements_opened, mode='lines', name='reengagements opened'))
fig11.add_trace(go.Scatter(x=week_of, y=reengagements_ct, mode='lines', name='reengagements clicked-through'))
fig11.update_layout(title='Email Open and Click-Through Rates', xaxis_title='Week of', yaxis_title='Number of emails')
pio.write_html(fig11, file='figures/fig11.html', auto_open=True)
```
{% include 2021-05-26-A-Drop-in-User-Engagement-fig11.html %}

Conclusions:
*   Click-through rate appears to be OK (no sharp decline to indicate a broken feature).
*   While open rates appear to increase, a further look into the chart indicates that our click-through rate drops after July 28th. This divergent behavior suggests that our email content might not be relevant enough to our users.

## Summary
From our analysis, we find that our mobile users are becoming less engaged with the platform. We also see email click-through rates decreasing, which tells us that our email reengagement and retention campaigns aren't going so well. Our next step is to reach out to the Engineering and Marketing teams to check for potential issues (e.g. product releases, marketing campaigns) that have contributed to the decline in engageent. We've successfully ruled-out major bugs (e.g., platform outages, regional differences, signup funnel issues) and have a good understanding of what steps to take next (A/B testing, local segmentation, feature roll-backs, improved email campaigns).

## Credit
The data and problem description was provided to us by Mode Analytics. The approach was based off the work of Jodi Zhang. A lot of time and energy was spent by all investigating each query and analysing goal and value metrics.
