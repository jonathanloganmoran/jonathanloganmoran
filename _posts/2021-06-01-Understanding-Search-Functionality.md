---
layout: post
title: Understanding Search Functionality
author: Jonathan Logan Moran
categories: portfolio
tags: plotly python SQL ipynb yammer case-studies feasibility-studies hypothesis-testing
permalink: /understanding-search-functionality
description: "Part 2 in a three-part study exploring the world of Yammer– a workplace communication platform. In this study, we investigate Yammer's search feature with the goal of making recommendations to the development team."
---

# Understanding Search Functionality
In this Part 2 of three case studies investigating Yammer–a social network for the workplace, we will be helping the product team determine priorities for their next development cycle. They are considering improving the site's search functionality, but first need our help deciding whether or not to work on the project in the first place. Thus, it's our job to consider the impact that search has on users, and see if there's any specific improvements that can be made. We'll be making recommendations to the development team, so it's best that we form quantitative hypotheses that seek to understand the general quality of an individual user's search experience.

_Let's get started..._

## Programming
In this study, we will be using a few common technologies:
*   SQL (SQLite) for data manipulation/exploration
*   Python (pandas) for data frames
*   Plotly/Jupyter notebook for interactive charts/data visualization


```python
# prerequisites
!pip install pandasql
!pip install plotly
```


```python
# data tools
import pandas as pd					# for querying pandas dataframes
import pandasql
from pandasql import sqldf
# plotting tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio					# output charts to HMTL
```


```python
#init_notebook_mode()
```

## Collecting our datasets
There are two tables that are relevant to this problem, `yammer_events` and `yammer_users`.

It is going to be crucial that we look at the following events in our events table:
*   `search_autocomplete`: this event is logged when a user clicks on a search option from autocomplete.
*   `search_run`: this event is logged when a user runs a search and sees the results page.
*   `search_click_X`: this event is logged when a user clicks on a search result `X`, which ranges from `1` to `10`. 


```python
events_path = '../src/yammer_events.csv'
```


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



## The problem
Here's a re-hash of the above: Yammer's product team is determining priorities for the next development cycle. They're considering improving the site's search functionality. Before they proceed, they want to know whether they should even work on search in the first place, and, if so, how they should modify it.


## Making a recommendation
In particular, you should seek to answer the following questions:
* Are users' search experiences generally good or bad?
* Is search worth working on at all?
* If search is worth working on, what, specifically, should be improved?

## Analysis
In order to form our hypotheses (by way of metrics we will investigate), we need to use the above questions to better understand the ultimate purpose of search at Yammer. For this study, we will consider a session to be a string of events that occur without a 10-minute break between any two events. That goes to say that if an active users fails to log an event within a 10-minute window, their session is considered over and the next engagement will mark a new session.

### _1. Search Use_
The first thing to understand is whether anyone even uses search at all.
### Search Use Over Time


```python
q = """
SELECT
  week_of,
  (with_autocompletes * 1.00 / sessions) * 100 AS perct_autocompletes,
  (with_runs * 1.00 / sessions) * 100 AS perct_runs
FROM (
      SELECT
        strftime('%Y-%m-%d', session_start, 'weekday 0', '-6 days') AS week_of,
        COUNT(*) sessions,
        COUNT(CASE WHEN autocompletes > 0 THEN session ELSE NULL END) AS with_autocompletes,
        COUNT(CASE WHEN runs > 0 THEN session ELSE NULL END) AS with_runs
      FROM (
            SELECT
              strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
              session,
              user_id,
              COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes,
              COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs,
              COUNT(CASE WHEN event_name LIKE 'search_click_result_%' THEN user_id ELSE NULL END) AS clicks
            FROM (
                  SELECT
                    e.*,
                    sessions.session,
                    sessions.session_start
                  FROM yammer_events e
                  LEFT JOIN (
                              SELECT
                                user_id,
                                session,
                                MIN(occurred_at) AS session_start,
                                MAX(occurred_at) AS session_end
                              FROM (
                                    SELECT
                                      intervals.*,
                                      CASE WHEN prev_event >= 10.0 THEN id
                                           WHEN prev_event IS NULL THEN id
                                           ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
                                           END AS session
                                    FROM (
                                          SELECT
                                            user_id,
                                            occurred_at,
                                            event_type,
                                            event_name,
                                            ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
                                            ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
                                            ROW_NUMBER() OVER () AS id
                                          FROM yammer_events
                                          WHERE event_type = 'engagement'
                                          ORDER BY user_id, occurred_at
                                    ) intervals
                                    WHERE prev_event >= 10.0
                                       OR next_event >= 10.0
                                       OR prev_event IS NULL
                                       OR next_event IS NULL
                              ) bounds
                              GROUP BY 1, 2
                  ) sessions
                  ON e.user_id = sessions.user_id
                  AND e.occurred_at >= sessions.session_start
                  AND e.occurred_at <= sessions.session_end
                  WHERE e.event_type = 'engagement'
            ) events
            GROUP BY 1, 2, 3
      ) counts
      GROUP BY 1
      ORDER BY 1
) counts
GROUP BY 1
ORDER BY 1
"""
```


```python
searches = pandasql.sqldf(q, globals())
```


```python
week_of = searches['week_of']
perct_autocompletes = searches['perct_autocompletes']
perct_runs = searches['perct_runs']
```


```python
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=week_of, y=perct_autocompletes, mode='lines', name='searches with autocompletes'))
fig1.add_trace(go.Scatter(x=week_of, y=perct_runs, mode='lines', name='searches with runs'))
fig1.update_layout(title='Search Rate by Week', xaxis_title='Week of', yaxis_title='Percent of sessions')
pio.write_html(fig1, file='2021-06-01-Understanding-Search-Functionality-fig1.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig1.html %}

### Search Use Per Session


```python
q = """
SELECT
  autocompletes,
  COUNT(*) AS sessions
FROM (
      SELECT
        strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
        session,
        user_id,
        COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes,
        COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs,
        COUNT(CASE WHEN event_name LIKE 'search_click_result_%' THEN user_id ELSE NULL END) AS clicks
      FROM (
            SELECT
              e.*,
              session,
              session_start
            FROM yammer_events e
            LEFT JOIN (
                        SELECT
                          user_id,
                          session,
                          MIN(occurred_at) AS session_start,
                          MAX(occurred_at) AS session_end
                        FROM (
                              SELECT
                                intervals.*,
                                CASE WHEN prev_event >= 10.0 THEN id
                                     WHEN prev_event IS NULL THEN id
                                     ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
                                     END AS session
                              FROM (
                                    SELECT
                                      user_id,
                                      occurred_at,
                                      event_type,
                                      event_name,
                                      ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
                                      ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
                                      ROW_NUMBER() OVER () AS id
                                    FROM yammer_events
                                    WHERE event_type = 'engagement'
                              ) intervals
                              WHERE prev_event >= 10.0
                                 OR next_event >= 10.0
                                 OR prev_event IS NULL
                                 OR next_event IS NULL
                        ) bounds
                        GROUP BY 1, 2
            ) sessions
            ON e.user_id = sessions.user_id
            AND e.occurred_at >= sessions.session_start
            AND e.occurred_at <= sessions.session_end
            WHERE e.event_type = 'engagement'
      ) events
      GROUP BY 1, 2, 3
) counts
WHERE autocompletes > 0
GROUP BY 1
ORDER BY 1
"""
```


```python
autocompletes = pandasql.sqldf(q, globals())
```


```python
num_autocompletes = autocompletes['autocompletes']
num_sessions = autocompletes['sessions']
```


```python
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=num_autocompletes, y=num_sessions, text=num_sessions, textposition='auto', name='sessions with autocomplete'))
fig2.update_yaxes(nticks=6)
fig2.update_layout(title='Number of Sessions with Autocompletes', xaxis_title='Autocompletes per session', yaxis_title='Number of sessions')
pio.write_html(fig2, file='2021-06-01-Understanding-Search-Functionality-fig2.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig2.html %}

Conclusions:
*   From the first chart, we can tell right off that bat that our users are running more `autocomplete` searches than they are with the search results page.
*   To be more specific, our `search_autocomplete` rate is ~23% while our `search_run` rate is roughly 8%.
*   In the second chart, it is clear that users are typically running autocomplete searches once or twice per session.
*   While the difference in search rates by type may seem alarming, we will assume that users are getting some value out of the autocomplete feature.

### _2. Search Frequency_
If users search a lot, it's likely that they're getting value out of the feature– with a major exception. If users search repeatedly within a short timeframe, it's likely that they're refining their terms because they were unable to find what they initially wanted.

### Search Runs Per Session


```python
q = """
SELECT
  runs,
  COUNT(*) AS sessions
FROM (
      SELECT
        strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
        session,
        user_id,
        COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes,
        COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs,
        COUNT(CASE WHEN event_name LIKE 'search_click_result_%' THEN user_id ELSE NULL END) AS clicks
      FROM (
            SELECT
              e.*,
              session,
              session_start
            FROM yammer_events e
            LEFT JOIN (
                        SELECT
                          user_id,
                          session,
                          MIN(occurred_at) AS session_start,
                          MAX(occurred_at) AS session_end
                        FROM (
                              SELECT
                                intervals.*,
                                CASE WHEN prev_event >= 10.0 THEN id
                                     WHEN prev_event IS NULL THEN id
                                     ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
                                     END AS session
                              FROM (
                                    SELECT
                                      user_id,
                                      strftime('%Y-%m-%d %H:%M:%S', occurred_at) AS occurred_at,
                                      event_type,
                                      event_name,
                                      ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
                                      ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
                                      ROW_NUMBER() OVER () AS id
                                    FROM yammer_events
                                    WHERE event_type = 'engagement'
                              ) intervals
                              WHERE prev_event >= 10.0
                                 OR next_event >= 10.0
                                 OR prev_event IS NULL
                                 OR next_event IS NULL
                        ) bounds
                        GROUP BY 1, 2
            ) sessions
            ON e.user_id = sessions.user_id
            AND e.occurred_at >= sessions.session_start
            AND e.occurred_at <= sessions.session_end
            WHERE e.event_type = 'engagement'
      ) events
      GROUP BY 1, 2, 3
) counts
WHERE runs > 0
GROUP BY 1
ORDER BY 1
"""
```


```python
runs = pandasql.sqldf(q, globals())
```


```python
num_runs = runs['runs']
num_sessions = runs['sessions']
```


```python
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=num_runs, y=num_sessions, text=num_sessions, textposition='auto', name='runs per session'))
fig3.update_layout(title='Number of Sessions with Runs', xaxis_title='Number of runs', yaxis_title='Number of sessions')
pio.write_html(fig3, file='2021-06-01-Understanding-Search-Functionality-fig3.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig3.html %}

Conclusions:
*   This chart looks at the number of searches (events defined as `search_run`) per session. This is an important metric to consider, since it appears that users are typically running multiple searches in a single session.
*   Since `search_run` (Yammer's results page) is a lesser used feature, this indicates that the search results might not be very good.

### _3. Clickthroughs_
If a user clicks many links in the search results, it's likely that she isn't having a great experience. However, the inverse is not necessarily true–clicking only one result does not imply a success. If the user clicks through one result, then refines their search, that's certainly not a great experience, so search frequently is probably a better way to understand that piece of the puzzle. Clickthroughs are, however, very useful in determining whether search rankings are good. If users frequently click low results or scroll to additional pages, then the ranking algorithm should probably be adjusted.

### Clickthroughs in Search Runs


```python
q = """
SELECT
  clicks,
  COUNT(*) AS sessions
FROM (
        SELECT
          strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
          session,
          user_id,
          COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes,
          COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs,
          COUNT(CASE WHEN event_name LIKE 'search_click_result_%' THEN user_id ELSE NULL END) AS clicks
        FROM (
              SELECT
                e.*,
                session,
                session_start
              FROM yammer_events e
              LEFT JOIN (
                          SELECT
                            user_id,
                            session,
                            MIN(occurred_at) AS session_start,
                            MAX(occurred_at) AS session_end
                          FROM (
                                SELECT
                                  intervals.*,
                                  CASE WHEN prev_event >= 10.0 THEN id
                                       WHEN prev_event IS NULL THEN id
                                       ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
                                       END AS session
                                FROM (
                                      SELECT
                                        user_id,
                                        occurred_at,
                                        event_type,
                                        event_name,
                                        ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
                                        ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
                                        ROW_NUMBER() OVER () AS id
                                      FROM yammer_events
                                      WHERE event_type = 'engagement'
                                ) intervals
                                WHERE prev_event >= 10.0
                                   OR next_event >= 10.0
                                   OR prev_event IS NULL
                                   OR next_event IS NULL
                          ) bounds
                          GROUP BY 1, 2
              ) sessions
              ON e.user_id = sessions.user_id
              AND e.occurred_at >= sessions.session_start
              AND e.occurred_at <= sessions.session_end
              WHERE e.event_type = 'engagement'
        ) events
        GROUP BY 1, 2, 3
) counts
WHERE runs > 0
GROUP BY 1
ORDER BY 1
"""
```


```python
clicks = pandasql.sqldf(q, globals())
```


```python
num_clicks = clicks['clicks']
num_sessions = clicks['sessions']
```


```python
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=num_clicks, y=num_sessions, text=num_sessions, textposition='auto', name='num clicks per session'))
fig4.update_layout(title='Clicks Per Session During Search Runs', xaxis_title='Clicks per session', yaxis_title='Number of sessions')
pio.write_html(fig4, file='2021-06-01-Understanding-Search-Functionality-fig4.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig4.html %}

### Average Clicks Per Search


```python
q = """
SELECT
  runs,
  ROUND(AVG(clicks), 2) AS avg_clicks
FROM (
      SELECT
        strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
        session,
        user_id,
        COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes,
        COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs,
        COUNT(CASE WHEN event_name LIKE 'search_click_result_%' THEN user_id ELSE NULL END) AS clicks
      FROM (
            SELECT
              e.*,
              session,
              session_start
            FROM yammer_events e
            LEFT JOIN (
                        SELECT
                          user_id,
                          session,
                          MIN(occurred_at) AS session_start,
                          MAX(occurred_at) AS session_end
                        FROM (
                              SELECT
                                intervals.*,
                                CASE WHEN prev_event >= 10.0 THEN id
                                     WHEN prev_event IS NULL THEN id
                                     ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
                                     END AS session
                              FROM (
                                    SELECT
                                      user_id,
                                      occurred_at,
                                      event_name,
                                      event_type,
                                      ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
                                      ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
                                      ROW_NUMBER() OVER () AS id
                                    FROM yammer_events
                                    WHERE event_type = 'engagement'
                              ) intervals
                              WHERE prev_event >= 10.0
                                 OR next_event >= 10.0
                                 OR prev_event IS NULL
                                 OR next_event IS NULL
                        ) bounds
                        GROUP BY 1, 2
            ) sessions
            ON e.user_id = sessions.user_id
            AND e.occurred_at >= sessions.session_start
            AND e.occurred_at <= sessions.session_end
            WHERE e.event_type = 'engagement'
      ) events
      GROUP BY 1, 2, 3
) counts
WHERE runs > 0
GROUP BY 1
ORDER BY 1
"""
```


```python
avg_clicks = pandasql.sqldf(q, globals())
```


```python
runs = avg_clicks['runs']
avg_clicks = avg_clicks['avg_clicks']
```


```python
fig5 = go.Figure()
fig5.add_trace(go.Bar(x=runs, y=avg_clicks, text=avg_clicks, textposition='auto', name='avg clicks'))
fig5.update_layout(title='Average Clicks Per Search', xaxis_title='Searches per session', yaxis_title='Average clicks per session')
pio.write_html(fig5, file='2021-06-01-Understanding-Search-Functionality-fig5.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig5.html %}

### Clicks by Search Result


```python
q = """
SELECT 
  r.*
FROM (
      SELECT
        CASE event_name
          WHEN 'search_click_result_1' THEN 1
          WHEN 'search_click_result_2' THEN 2
          WHEN 'search_click_result_3' THEN 3
          WHEN 'search_click_result_4' THEN 4
          WHEN 'search_click_result_5' THEN 5
          WHEN 'search_click_result_6' THEN 6
          WHEN 'search_click_result_7' THEN 7
          WHEN 'search_click_result_8' THEN 8
          WHEN 'search_click_result_9' THEN 9
          WHEN 'search_click_result_10' THEN 10
          END AS result,
        COUNT(user_id) AS clicks
      FROM yammer_events
      GROUP BY 1
) r
WHERE r.result IS NOT NULL
"""
```


```python
results = pandasql.sqldf(q, globals())
```


```python
result_num = results['result']
num_clicks = results['clicks']
```


```python
fig6 = go.Figure()
fig6.add_trace(go.Bar(x=result_num, y=num_clicks, text=num_clicks, textposition='auto', name='Clicks per result'))
fig6.update_layout(title='Clicks by Result Order', xaxis_title='Search result order', yaxis_title='Number of clicks')
pio.write_html(fig6, file='2021-06-01-Understanding-Search-Functionality-fig6.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig6.html %}

### Search Retention


```python
q = """
WITH intervals AS (
SELECT
  user_id,
  occurred_at,
  event_name,
  event_type,
  ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
  ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
  ROW_NUMBER() OVER () AS id
FROM yammer_events
WHERE event_type = 'engagement'
),
bounds AS (
SELECT
  user_id,
  occurred_at,
  CASE WHEN prev_event >= 10.0 THEN id
       WHEN prev_event IS NULL THEN id
       ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
      END AS session
FROM intervals
WHERE prev_event >= 10.0
  OR next_event >= 10.0
  OR prev_event IS NULL
  OR next_event IS NULL
),
sessions AS (
SELECT
  user_id,
  session,
  MIN(occurred_at) AS session_start,
  MAX(occurred_at) AS session_end
FROM bounds
GROUP BY 1, 2
),
first AS (
SELECT
  user_id,
  MIN(occurred_at) AS first_run
FROM yammer_events
WHERE event_name = 'search_run'
GROUP BY 1
),
events AS (
SELECT
  e.*,
  sessions.session,
  sessions.session_start,
  first.first_run
FROM yammer_events e
JOIN first
ON e.user_id = first.user_id
LEFT JOIN sessions
ON e.user_id = sessions.user_id
AND e.occurred_at >= sessions.session_start
AND e.occurred_at <= sessions.session_end
AND sessions.session_start <= datetime(first.first_run, '1 MONTH')
WHERE e.event_type = 'engagement'
),
counts AS (
SELECT
  strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
  session,
  user_id,
  first_run,
  COUNT(CASE WHEN event_name = 'search_run' THEN user_id ELSE NULL END) AS runs
FROM events
GROUP BY 1, 2, 3, 4
),
runs AS (
SELECT
  user_id,
  COUNT(*) AS searches
FROM counts
WHERE runs > 0
GROUP BY 1
ORDER BY 1
)
SELECT
  searches,
  COUNT(*) AS users
FROM runs
GROUP BY 1
ORDER BY 1
"""
```


```python
run_retention = pandasql.sqldf(q, globals())
```


```python
searches = run_retention['searches']
users = run_retention['users']
```


```python
fig7 = go.Figure()
fig7.add_trace(go.Bar(x=searches, y=users, text=users, textposition='auto', name='sessions with search runs'))
fig7.update_layout(title="Sessions with Search Runs Month After Users' First Search", xaxis_title='Number of sessions with search runs', yaxis_title='Number of users')
pio.write_html(fig7, file='2021-06-01-Understanding-Search-Functionality-fig7.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig7.html %}

Conclusions:
*   This metric (our click-through rates) has an important role in determining the effectiveness of our search feature.
*   The first chart in this segment, _Clicks Per Session During Search Runs_, reveals a lot about the quality of our search results. In sessions where users do search, they almost never click any of the results.
*   Our next chart, _Average Clicks Per Search_, we determine that more searches in a given session do not lead to many more clicks, on average.
*   _Clicks by Result Order_ surprises us, as we expected our search ranking algorithm to perform better. That is, we expected the top results returned to be the most clicked when in reality our users appear to have an even distribution of clicks across our search results.
*   Lastly, users rarely run full searches again in the month following a session with a `search_run`.

### _4. Autocomplete Clickthroughs_


```python
q = """
WITH intervals AS (
SELECT
  user_id,
  occurred_at,
  event_type,
  event_name,
  ROUND((JULIANDAY(occurred_at) - JULIANDAY(LAG(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at))) * 1440) AS prev_event,
  ROUND((JULIANDAY(LEAD(occurred_at, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)) - JULIANDAY(occurred_at)) * 1440) AS next_event,
  ROW_NUMBER() OVER () AS id
FROM yammer_events
WHERE event_type = 'engagement'
),
bounds AS (
SELECT
  user_id,
  occurred_at,
  CASE WHEN prev_event >= 10.0 THEN id
       WHEN next_event IS NULL THEN id
       ELSE LAG(id, 1) OVER (PARTITION BY user_id ORDER BY occurred_at)
       END AS session
FROM intervals
WHERE prev_event >= 10.0
   OR next_event >= 10.0
   OR prev_event IS NULL
   OR next_event IS NULL
),
sessions AS (
SELECT
  user_id,
  session,
  MIN(occurred_at) AS session_start,
  MAX(occurred_at) AS session_end
FROM bounds
GROUP BY 1, 2
),
first AS (
SELECT
  user_id,
  MIN(occurred_at) AS first_autocomplete
FROM yammer_events
WHERE event_name = 'search_autocomplete'
GROUP BY 1
),
events AS (
SELECT
  e.*,
  sessions.session,
  sessions.session_start,
  first.first_autocomplete
FROM yammer_events e
JOIN first
ON e.user_id = first.user_id
LEFT JOIN sessions
ON e.user_id = sessions.user_id
AND e.occurred_at >= sessions.session_start
AND e.occurred_at <= sessions.session_end
AND sessions.session_start <= datetime(first.first_autocomplete, '1 MONTH')
WHERE e.event_type = 'engagement'
),
counts AS (
SELECT
  strftime('%Y-%m-%d %H:%M:%S', session_start) AS session_start,
  session,
  user_id,
  first_autocomplete,
  COUNT(CASE WHEN event_name = 'search_autocomplete' THEN user_id ELSE NULL END) AS autocompletes
FROM events
GROUP BY 1, 2, 3, 4
),
searches AS (
SELECT
  user_id,
  COUNT(*) AS searches
FROM counts
WHERE autocompletes > 0
GROUP BY 1
)
SELECT
  searches,
  COUNT(*) AS users
FROM searches
GROUP BY 1
ORDER BY 1
"""
```


```python
autocompletes = pandasql.sqldf(q, globals())
```


```python
searches = autocompletes['searches']
users = autocompletes['users']
```


```python
fig8 = go.Figure()
fig8.add_trace(go.Bar(x=searches, y=users, text=users, textposition='auto', name='sessions with autocompletes'))
fig8.update_layout(title="Autocomplete Sessions Month After Users' First Search", xaxis_title='Number of sessions with autocompletes', yaxis_title='Number of users')
pio.write_html(fig8, file='2021-06-01-Understanding-Search-Functionality-fig8.html', auto_open=True)
```
{% include 2021-06-01-Understanding-Search-Functionality-fig8.html %}

Conclusions:
*   Our last figure, _Autocomplete Sessions Month After Users' First Search_, tells us that users continue to use the autocomplete search feature following a session with an autocomplete search. In comparison to the previous chart, our autocomplete feature appears to have a higher rate of returning use than our search results.

## Summary
After looking at the data more thoroughly, we can safely conclude that Yammer's autocomplete feature is performing reasonably well. From our analysis, we determine that the best place to start with improvements is our search results page, which appears to be performing poorly. Users seem to have very low click-through rates when running full searches, indicating that they are not finding the result they are looking for. Since Yammer has a goal metric of providing the most value out of every feature, we recommend that the development team begin the task of improving our search ranking algorithm.

## Credit
The data and problem description used in this study was provided by Mode Analytics.