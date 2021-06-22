---
layout: post
title: My First Working Plotly Post
author: Jonathan Logan Moran
categories: tests
tags: plotly python ipynb jekyll test first
permalink: /my-first-working-plotly-post
description: "This notebook addresses the `Liquid::SyntaxError` that occurs with Plotly.js in Markdown files by eliminating in-notebook rendering of Plotly charts (using `pio.write_html`) and removing Plotly.js DOM (no `init_notebook_mode()`)."
---

# Testing Plotly
**UPDATE 2021-06-21:** Successfully rendering Plotly charts in Markdown posts! For more info on how this is possible, check out [my post](https://stackoverflow.com/questions/68061995/liquidsyntaxerror-with-plotly-in-markdown-files-using-jekyll) on Stack Overflow.
> This notebook reproduces a Liquid syntax error when downloading `.ipynb` and converting notebooks with Plotly charts to `.md` for a Jekyll website.

## Import dependencies


```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
```

## Plotly `.ipynb` configuration
**UPDATE:** The following lines of code *must* be ommitted in Jupyter/Google Colab notebook. Doing so will prevent Plotly from including the `Plotly.js` library in your notebook's output file. If ran, this adds ~1000 lines of code to your output file and leads to a `Liquid::SyntaxError` mentioned in the Stack Overflow post above.

> ```python
> # from plotly.offline import init_notebook_mode, iplot
> # init_notebook_mode(connected=False)
> # pio.renderers.default = 'jupyterlab' 	# loads plotly.js bundle into notebook (AVOID)
> ```

Now onto the exciting part...

## Bar Chart with Plotly Express
Here's a quick demo of a Plotly Express Bar chart:

> ```python
> data_canada = px.data.gapminder().query("country == 'Canada'")
> ```
>
>
> ```python
> fig = go.Figure()
> fig.add_trace(go.Bar(x=data_canada['year'], y=data_canada['pop']))
> fig.update_layout(title="Canada's Population Per Year", xaxis_title='Year', yaxis_title='Population')
> ```

**The workaround:** Adding the following line of code is *key* to preparing your Plotly charts in a Markdown-friendly format. This will export each figure to HTML, and will also avoid rendering your Plotly figures directly in the notebook file.

> ```python
> pio.write_html(fig, file='../assets/figures/2021-06-20/fig.html', auto_open=True)
> ```

The next and final step is to add the following line of code to your Markdown file in the desired location you wish to display your Plotly chart:

> {% raw %}
> {% include 2021-06-20-fig.html %}
> {% endraw %}

Note: you must include the figure's HTML file in your Jekyll `_includes` folder (subdirectories are not permitted).

{% include 2021-06-20-fig.html %}

**Congratulations!** You did it. This simple workaround lets you render Plotly figures in HTML format, a perfect solution to displaying your hard work on any Jekyll site, blog, portfolio, etc.
