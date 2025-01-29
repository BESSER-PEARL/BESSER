from typing import Union

from besser.BUML.metamodel.structural import Model, NamedElement, Element

MANDATORY = 'mandatory'
OPTIONAL = 'optional'
OR = 'or'
ALTERNATIVE = 'alternative'


class FeatureValue(Element):

    def __init__(self, t: str, values: list = None, min: float = None, max: float = None):
        if ((min or max) and not values) or (not min and not max):
            if t == 'int':
                if values and any(map(lambda x: not isinstance(x, int), values)):
                    raise ValueError('Value must be an integer')
            if t == 'float':
                if values and any(map(lambda x: not isinstance(x, float), values)):
                    raise ValueError('Value must be a float')
            if t == 'str':
                if values and any(map(lambda x: not isinstance(x, str), values)):
                    raise ValueError(' Value must be a string')
        else:
            raise ValueError('Invalid arguments')
        self.type: str = t
        self.values: list = values
        self.min: int = min
        self.max: int = max

    def __eq__(self, other):
        if type(other) is type(self):
            return self.type == other.type and self.values == other.values and self.min == other.min and self.max == other.max
        else:
            return False


class Feature(NamedElement):

    @staticmethod
    def duplicate(f: 'Feature', parent: 'Feature' = None, min: int = 1, max: int = 1) -> 'Feature':
        new_f = Feature(f.name, min=min, max=max, value=f.value)
        new_f.parent = parent
        for children_group in f.children_groups:
            new_f.children_groups.append(children_group.duplicate(new_f))
        return new_f

    def __init__(self, name: str, min: int = 1, max: int = 1, value: FeatureValue = None):
        super().__init__(name)
        if min > max or min < 1:
            raise ValueError(f'Error in {name}: 0 < min < max')
        self.min: int = min
        self.max: int = max
        self.value: FeatureValue = value
        self.parent: Feature = None
        self.children_groups: list[FeatureGroup] = []

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name and self.min == other.min and self.max == other.max and self.value == other.value and self.children_groups == other.children_groups
        else:
            return False

    def to_json(self):
        d = []
        for children_group in self.children_groups:
            g = {'kind': children_group.kind, 'features': []}
            d.append(g)
            for feature in children_group.features:
                g['features'].append(feature.to_json())
        return {self.name: d}

    def mandatory(self, child: 'Feature') -> 'Feature':
        if child.parent is not None:
            raise ValueError(f'Feature {child.name} cannot be a child of {self.name}. It has feature {child.parent.name} as parent.')
        self.children_groups.append(FeatureGroup(MANDATORY, [child]))
        child.parent = self
        return self

    def optional(self, child: 'Feature') -> 'Feature':
        if child.parent is not None:
            raise ValueError(f'Feature {child.name} cannot be a child of {self.name}. It has feature {child.parent.name} as parent.')
        self.children_groups.append(FeatureGroup(OPTIONAL, [child]))
        child.parent = self
        return self

    def alternative(self, children: list['Feature']) -> 'Feature':
        for child in children:
            if child.parent is not None:
                raise ValueError(f'Feature {child.name} cannot be a child of {self.name}. It has feature {child.parent.name} as parent.')
            child.parent = self
        self.children_groups.append(FeatureGroup(ALTERNATIVE, children))
        return self

    def or_(self, children: list['Feature']) -> 'Feature':
        for child in children:
            if child.parent is not None:
                raise ValueError(f'Feature {child.name} cannot be a child of {self.name}. It has feature {child.parent.name} as parent.')
            child.parent = self
        self.children_groups.append(FeatureGroup(OR, children))
        return self

    def get_depth(self, depth: int = 0) -> int:
        max_depth = depth
        for children_group in self.children_groups:
            for child in children_group.features:
                child_depth = child.get_depth(depth+1)
                if child_depth > max_depth:
                    max_depth = child_depth
        return max_depth


class FeatureGroup(Element):

    def __init__(self, kind: str, features: list[Feature] = None):
        if features is None:
            features = []
        if (kind == MANDATORY or kind == OPTIONAL) and len(features) > 1:
            raise ValueError(f'{kind} has more than 1 feature')
        if (kind == ALTERNATIVE or kind == OR) and len(features) < 2:
            raise ValueError(f'{kind} has less than 2 features')

        self.features: list[Feature] = features
        self.kind: str = kind

    def __eq__(self, other):
        if type(other) is type(self):
            return self.kind == other.kind and self.features == other.features
        else:
            return False

    def duplicate(self, parent: Feature) -> 'FeatureGroup':
        new_children: list[Feature] = []
        for f in self.features:
            new_children.append(Feature.duplicate(f, parent, min=f.min, max=f.max))
        return FeatureGroup(self.kind, new_children)


class FeatureConfiguration(Element):

    def __init__(self, feature: Feature, value: Union[int, float, str] = None):
        self.feature: Feature = feature
        self.value: Union[int, float, str] = value
        self.parent: FeatureConfiguration = None
        self.children: list[FeatureConfiguration] = []

    def to_json(self):
        c = []
        for child in self.children:
            c.append(child.to_json())
        if c:
            if len(c) == 1:
                return {self.feature.name: c[0]}
            return {self.feature.name: c}
        elif self.value is not None:
            return {self.feature.name: self.value}
        else:
            return self.feature.name

    def add_child(self, child: 'FeatureConfiguration') -> None:
        child.parent = self
        self.children.append(child)

    def add_children(self, children: list['FeatureConfiguration']) -> None:
        for child in children:
            child.parent = self
        self.children.extend(children)

    def get_child(self, name: str) -> 'FeatureConfiguration':
        child = [c for c in self.children if c.feature.name == name]
        if len(child) > 1:
            raise ValueError(f"Feature {self.feature.name} has {len(child)} children with the name {name}. Make sure there are no more than one children with the same name")
        if len(child) == 0:
            return None
        return child[0]

    def get_children(self, name: str) -> list['FeatureConfiguration']:
        return [c for c in self.children if c.feature.name == name]

    def get_depth(self, depth: int = 0) -> int:
        max_depth = depth
        for child in self.children:
            child_depth = child.get_depth(depth+1)
            if child_depth > max_depth:
                max_depth = child_depth
        return max_depth


class FeatureModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.root_feature: Feature = None

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name and self.root_feature == other.root_feature
        else:
            return False

    def root(self, feature: Feature) -> 'FeatureModel':
        self.root_feature = feature
        return self

    def duplicate(self, min: int = 1, max: int = 1) -> Feature:
        return Feature.duplicate(f=self.root_feature, min=min, max=max)
