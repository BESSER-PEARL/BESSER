from datetime import datetime
{% for class in classes %}
class {{ class.name }}{% if class.parents() %}({% for parent in class.parents() %}{{parent.name}}{{ ", " if not loop.last else "" }}{% endfor %}){% endif %}:
    def __init__(self{% for attribute in class.all_attributes() %}, {{ attribute.name }}: {{ attribute.type.name }}{% endfor %}
                 {%- for end in class.association_ends() if end.is_navigable == True %}, {{ end.name }}: {% if end.multiplicity.max > 1 %}set[{{ end.type.name }}]
                 {%- else %}{{ end.type.name }}{%- endif %}{%- endfor %}):
        {% if class.inherited_attributes() %}super().__init__({% for attr in class.inherited_attributes() %}{{attr.name}}{{ ", " if not loop.last else "" }}{% endfor %})
        {% endif -%}
        {% for attribute in class.attributes %}self.{{ attribute.name }} = {{ attribute.name }}
        {% endfor -%}
    {% for attribute in class.attributes %}
    @property
    def {{ attribute.name }}(self) -> {{ attribute.type.name }}:
        return self.__{{ attribute.name }}
    
    @{{ attribute.name }}.setter
    def {{ attribute.name }}(self, {{ attribute.name }}: {{ attribute.type.name }}):
        self.__{{ attribute.name }} = {{ attribute.name }}
    {% endfor -%}
{% endfor %}