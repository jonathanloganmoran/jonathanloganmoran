---
layout: post
title: My First Working Plotly Post
author: Jonathan Logan Moran
categories: tests
tags: plotly python test first
permalink: /my-first-working-plotly-post
description: "Fixed `Liquid::SyntaxError` by eliminating in-notebook rendering of Plotly charts (using `pio.write_html`) and removing Plotly.js DOM (no `init_notebook_mode()`)."
---

# Testing Plotly
This notebook reproduces a Liquid syntax error when downloading `.ipynb` and converting notebooks with Plotly charts to `.md` for a Jekyll website.

## Import dependencies


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
```

## Plotly `.ipynb` configuration


```python
# redundant as I already have jupyterlab plotly-extension installed
# init_notebook_mode(connected=False)
```


```python
# load plotly.js bundle into notebook
# pio.renderers.default = 'jupyterlab'
```

## Bar Chart with Plotly Express


```python
data_canada = px.data.gapminder().query("country == 'Canada'")
```


```python
fig = go.Figure()
fig.add_trace(go.Bar(x=data_canada['year'], y=data_canada['pop']))
fig.update_layout(title='', xaxis_title='', yaxis_title='')
pio.write_html(fig, file='../assets/figures/2021-06-20/fig.html', auto_open=True)
```

{% include 2021-06-20-fig.html %}
