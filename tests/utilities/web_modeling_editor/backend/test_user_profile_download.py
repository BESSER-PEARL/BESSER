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
            'className': 'User', 'attributes': [],
        }
        for i in range(1, count + 1)
    }
    response = client.post('/besser_api/generate-output', json={
        'title': 'Profiles', 'generator': 'jsonobject',
        'model': {'type': 'UserDiagram', 'elements': elements, 'relationships': {}},
    })
    assert response.status_code == 200, response.text
    assert response.content
    if count == 1:
        document = response.json()
        assert document['model']['class'] == 'User'
    else:
        assert response.headers['content-type'].startswith('application/zip')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            assert len(names) == count
            assert all(json.loads(archive.read(name))['model']['class'] == 'User' for name in names)
