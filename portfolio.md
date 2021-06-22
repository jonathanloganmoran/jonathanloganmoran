---
layout: default
---

# My Work
_Note:_ this page is a work-in-progress and does not reflect the contents of my actual portfolio. Please [contact me](mailto:jonathan.moran107@gmail.com) if you would like to view my work offline while I am sorting out the contents of this site.
{% for category in site.categories %}
   <h3>{{ category[0] }}</h3>
   <ul>
     {% for post in category[1] %}
       <li><a href="{{ post.url }}">{{ post.title }}</a></li>
     {% endfor %}
   </ul>
 {% endfor %}
