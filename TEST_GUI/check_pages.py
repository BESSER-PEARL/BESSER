import json

with open('react_output_test/frontend/src/data/ui_components.json', encoding='utf-8') as f:
    data = json.load(f)

pages = data.get('pages', [])
print(f'\n✅ Total pages in React output: {len(pages)}\n')

for i, page in enumerate(pages, 1):
    print(f"{i}. ID: {page['id']}")
    print(f"   Name: {page['name']}")
    print(f"   Route: {page.get('route_path', 'N/A')}")
    print(f"   Is Main: {page.get('is_main', False)}")
    print()
