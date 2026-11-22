"""User-profile JSON downloads exercise the real HTTP response body."""

import io
import json
import zipfile

import pytest


@pytest.mark.parametrize('count', [1, 2])
def test_user_profile_download_contains_generated_json(client, count):
    elements = {
        f'user-{i}': {
            'id': f'user-{i}', 'type': 'UserModelName', 'name': f'user_{i}',
            'className': 'User', 'attributes': [f'name-{i}'],
            'personalization': {'presentation': {'language': f'language_{i}'},
                                'gui': {'modality': {'readAloud': True}}},
        }
        for i in range(1, count + 1)
    }
    elements.update({
        f'name-{i}': {'id': f'name-{i}', 'type': 'UserModelAttribute', 'owner': f'user-{i}',
                      'name': f'name = Person{i}',
                      'personalization': {'content': {'adaptContentToUserProfile': True}}}
        for i in range(1, count + 1)
    })
    response = client.post('/besser_api/generate-output', json={
        'title': 'Profiles', 'generator': 'jsonobject',
        'model': {'type': 'UserDiagram', 'elements': elements, 'relationships': {}},
    })
    assert response.status_code == 200, response.text
    assert response.content
    if count == 1:
        document = response.json()
        assert document['model']['class'] == 'User'
        assert document['personalizationRules'][0]['rules']['gui']['modality']['readAloud'] is True
        assert document['personalizationRules'][1]['source'] == 'attribute'
    else:
        assert response.headers['content-type'].startswith('application/zip')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            assert len(names) == count
            for name in names:
                document = json.loads(archive.read(name))
                assert document['model']['class'] == 'User'
                expected = 'language_1' if 'Person1' in name else 'language_2'
                assert document['personalizationRules'][0]['rules']['presentation']['language'] == expected
                assert document['personalizationRules'][1]['rules']['content']['adaptContentToUserProfile'] is True


def test_download_keeps_rules_captured_before_editor_merges_profile_boxes(client):
    captured = [
        {'source': 'box', 'elementId': 'before-merge-1', 'className': 'Personal_Information',
         'rules': {'presentation': {'font': 'serif'}}},
        {'source': 'box', 'elementId': 'before-merge-2', 'className': 'Personal_Information',
         'rules': {'gui': {'presentation': {'size': 20}}}},
    ]
    response = client.post('/besser_api/generate-output', json={
        'title': 'Profile', 'generator': 'jsonobject',
        'model': {'type': 'UserDiagram', 'elements': {
            'root': {'id': 'root', 'type': 'UserModelName', 'name': 'user_1', 'className': 'User',
                     'attributes': [], 'profilePersonalizationRules': captured},
        }, 'relationships': {}},
    })
    assert response.status_code == 200, response.text
    assert response.json()['personalizationRules'] == captured
