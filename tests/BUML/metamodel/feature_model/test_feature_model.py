import pytest

from besser.BUML.metamodel.feature_model.feature_model import FeatureModel, Feature, FeatureValue, FeatureGroup, \
    MANDATORY, OPTIONAL, ALTERNATIVE, OR

INF = 999


def test_feature_value():
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='int', values=[1, 2, 3], min=1)
    assert "Invalid arguments" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='int', values=[1, 2, 3], max=1)
    assert "Invalid arguments" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='int', values=[1, 2, 3], min=1, max=1)
    assert "Invalid arguments" in str(excinfo.value)

    # This should work
    feature_value = FeatureValue(t='int', values=[1, 2, 3])
    feature_value = FeatureValue(t='float', values=[1.0, 2.0, 3.0])
    feature_value = FeatureValue(t='str', values=['a', 'b', 'c'])

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='str', values=[1, 2, 3])
    assert "Value must be a string" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='int', values=['a', 'b', 'c'])
    assert "Value must be an integer" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_value = FeatureValue(t='float', values=['a', 'b', 'c'])
    assert "Value must be a float" in str(excinfo.value)


def test_feature_cardinality():
    # This should work
    feature = Feature('feature', min=1, max=1)
    feature = Feature('feature', min=1, max=2)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature = Feature('feature', min=2, max=1)
    assert f'Error in {feature.name}: 0 < min < max' in str(excinfo.value)


def test_mandatory():
    feature1 = Feature('feature1')
    feature2 = Feature('feature2')
    child = Feature('child')

    # This should work
    feature1.mandatory(child)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature2.mandatory(child)
    assert f'Feature {child.name} cannot be a child of {feature2.name}. It has feature {child.parent.name} as parent.' in str(excinfo.value)


def test_optional():
    feature1 = Feature('feature1')
    feature2 = Feature('feature2')
    child = Feature('child')

    # This should work
    feature1.optional(child)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature2.optional(child)
    assert f'Feature {child.name} cannot be a child of {feature2.name}. It has feature {child.parent.name} as parent.' in str(
        excinfo.value)


def test_alternative():
    feature1 = Feature('feature1')
    feature2 = Feature('feature2')
    child1 = Feature('child1')
    child2 = Feature('child2')

    # This should work
    feature1.alternative([child1, child2])

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature2.alternative([child1, child2])
    assert f'Feature {child1.name} cannot be a child of {feature2.name}. It has feature {child1.parent.name} as parent.' in str(excinfo.value)


def test_or():
    feature1 = Feature('feature1')
    feature2 = Feature('feature2')
    child1 = Feature('child1')
    child2 = Feature('child2')

    # This should work
    feature1.or_([child1, child2])

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature2.or_([child1, child2])
    assert f'Feature {child1.name} cannot be a child of {feature2.name}. It has feature {child1.parent.name} as parent.' in str(excinfo.value)


def test_feature_group():
    feature1 = Feature('feature1')
    feature2 = Feature('feature2')

    # This should work
    feature_group = FeatureGroup(kind=MANDATORY, features=[feature1])
    feature_group = FeatureGroup(kind=OPTIONAL, features=[feature1])
    feature_group = FeatureGroup(kind=ALTERNATIVE, features=[feature1, feature2])
    feature_group = FeatureGroup(kind=OR, features=[feature1, feature2])

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_group = FeatureGroup(kind=MANDATORY, features=[feature1, feature2])
    assert f'{MANDATORY} has more than 1 feature' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_group = FeatureGroup(kind=OPTIONAL, features=[feature1, feature2])
    assert f'{OPTIONAL} has more than 1 feature' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_group = FeatureGroup(kind=ALTERNATIVE, features=[feature1])
    assert f'{ALTERNATIVE} has less than 2 features' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        feature_group = FeatureGroup(kind=OR, features=[feature1])
    assert f'{OR} has less than 2 features' in str(excinfo.value)


def test_duplicate_feature_model():
    feature_model1 = (FeatureModel('feature_model').root(
        Feature('feature1')
        .mandatory(Feature('feature2'))
        .optional(Feature('feature2'))
        .alternative([
            Feature('feature3'),
            Feature('feature4')
        ])
        .or_([
            Feature('feature5'),
            Feature('feature6')
            .mandatory(Feature('feature7', min=2, max=10, value=FeatureValue('int', values=[1, 2, 3])))
        ])
    ))
    feature_model2 = FeatureModel('feature_model').root(feature_model1.duplicate())
    assert feature_model1 == feature_model2
    feature_model2.root_feature.mandatory(Feature('feature8'))
    assert feature_model1 != feature_model2
