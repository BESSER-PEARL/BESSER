from datetime import datetime
{% for class in classes %}class {{ class.name }}:
    def __init__(self{% for attribute in class.all_attributes() %}, {{ attribute.name }}:{{ attribute.type.name }}{% endfor %}):
        {% for attribute in class.attributes %}self.{{ attribute.name }} = {{ attribute.name }}
        {% endfor %}
{% endfor %}

