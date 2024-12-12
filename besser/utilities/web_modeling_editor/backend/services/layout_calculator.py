def calculate_center_point(bounds):
    """Calculate the center point of an element based on its bounds."""
    return {
        'x': bounds['x'] + bounds['width'] / 2,
        'y': bounds['y'] + bounds['height'] / 2
    }

def determine_connection_direction(source_bounds, target_bounds):
    """Determine the best connection directions between two elements."""
    source_center = calculate_center_point(source_bounds)
    target_center = calculate_center_point(target_bounds)
    
    dx = target_center['x'] - source_center['x']
    dy = target_center['y'] - source_center['y']
    
    # Si les éléments sont principalement alignés horizontalement
    if abs(dx) > abs(dy):
        if dx > 0:
            return "Right", "Left"  # source à gauche du target
        else:
            return "Left", "Right"  # source à droite du target
    # Si les éléments sont principalement alignés verticalement
    else:
        if dy > 0:
            return "Down", "Up"    # source au-dessus du target
        else:
            return "Up", "Down"    # source en-dessous du target

def calculate_connection_points(element_bounds, direction):
    """Calculate connection point based on direction."""
    if direction == "Right":
        return {
            'x': element_bounds['x'] + element_bounds['width'],
            'y': element_bounds['y'] + (element_bounds['height'] / 2)
        }
    elif direction == "Left":
        return {
            'x': element_bounds['x'],
            'y': element_bounds['y'] + (element_bounds['height'] / 2)
        }
    elif direction == "Up":
        return {
            'x': element_bounds['x'] + (element_bounds['width'] / 2),
            'y': element_bounds['y']
        }
    else:  # Down
        return {
            'x': element_bounds['x'] + (element_bounds['width'] / 2),
            'y': element_bounds['y'] + element_bounds['height']
        }

def calculate_path_points(source_point, target_point, source_dir, target_dir):
    """Calculate intermediate points for the relationship path."""
    points = [source_point]
    offset = 30  # Distance minimale pour les détours
    
    # Calculer les points intermédiaires en fonction des directions
    if source_dir == "Right" and target_dir == "Left":
        mid_x = (source_point['x'] + target_point['x']) / 2
        points.extend([
            {'x': mid_x, 'y': source_point['y']},
            {'x': mid_x, 'y': target_point['y']}
        ])
    elif source_dir == "Left" and target_dir == "Right":
        mid_x = (source_point['x'] + target_point['x']) / 2
        points.extend([
            {'x': mid_x, 'y': source_point['y']},
            {'x': mid_x, 'y': target_point['y']}
        ])
    elif source_dir == "Down" and target_dir == "Up":
        mid_y = (source_point['y'] + target_point['y']) / 2
        points.extend([
            {'x': source_point['x'], 'y': mid_y},
            {'x': target_point['x'], 'y': mid_y}
        ])
    elif source_dir == "Up" and target_dir == "Down":
        mid_y = (source_point['y'] + target_point['y']) / 2
        points.extend([
            {'x': source_point['x'], 'y': mid_y},
            {'x': target_point['x'], 'y': mid_y}
        ])
    elif source_dir in ["Right", "Left"] and target_dir in ["Up", "Down"]:
        points.extend([
            {'x': target_point['x'], 'y': source_point['y']}
        ])
    elif source_dir in ["Up", "Down"] and target_dir in ["Right", "Left"]:
        points.extend([
            {'x': source_point['x'], 'y': target_point['y']}
        ])
    
    points.append(target_point)
    return points

def calculate_relationship_bounds(path_points):
    """Calculate the bounding box that contains all path points with padding."""
    x_coords = [p['x'] for p in path_points]
    y_coords = [p['y'] for p in path_points]
    
    padding = 10  # Ajouter un peu d'espace autour du chemin
    min_x = min(x_coords) - padding
    max_x = max(x_coords) + padding
    min_y = min(y_coords) - padding
    max_y = max(y_coords) + padding
    
    return {
        'x': min_x,
        'y': min_y,
        'width': max_x - min_x,
        'height': max_y - min_y
    }