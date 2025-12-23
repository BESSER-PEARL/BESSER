"""
This module generates React code using Jinja2 templates based on BUML models.
"""
from __future__ import annotations

import json
import os
import shutil
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.gui import (
    Button,
    DataList,
    EmbeddedContent,
    Form,
    GUIModel,
    Image,
    InputField,
    Link,
    Menu,
    MenuItem,
    Text,
    ViewComponent,
    ViewContainer,
)
from besser.BUML.metamodel.gui.dashboard import (
    AgentComponent,
    BarChart,
    FieldColumn,
    LookupColumn,
    ExpressionColumn,
    LineChart,
    MetricCard,
    PieChart,
    RadarChart,
    RadialBarChart,
    Table,
)
from besser.BUML.metamodel.gui.events_actions import (
    Create,
    Delete,
    Event,
    Parameter,
    Read,
    Transition,
    Update,
)
from besser.BUML.metamodel.structural import DomainModel, Enumeration
from besser.generators import GeneratorInterface


class ReactGenerator(GeneratorInterface):
    """
    Generates React code based on BUML and GUI models. It walks all template files and renders
    them with the serialized GUI metadata so the resulting React project mirrors the modelling editor.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel, output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
            variable_start_string="[[",
            variable_end_string="]]",
        )
        self._style_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._raw_style_entries: List[Dict[str, Any]] = list(getattr(self.gui_model, "_style_entries", []))

    def generate(self):
        """
        Generates React TS code based on the provided BUML and GUI models.
        Generates all files from the templates directory, preserving structure and file names (removing .j2 extension).
        Only generates chart component files that are actually used in the GUI model.
        """

        context = self._build_generation_context()
        used_component_types = self._get_used_component_types()

        # Map component types to their template file patterns
        component_template_map = {
            'LineChart': ['LineChartComponent.tsx.j2'],
            'BarChart': ['BarChartComponent.tsx.j2'],
            'PieChart': ['PieChartComponent.tsx.j2'],
            'RadarChart': ['RadarChartComponent.tsx.j2'],
            'RadialBarChart': ['RadialBarChartComponent.tsx.j2'],
            'Table': ['TableComponent.tsx.j2', 'TableComponent.css.j2', 'ColumnFilter.tsx.j2', 'ColumnSort.tsx.j2'],
            'MetricCard': ['MetricCardComponent.tsx.j2'],
        }

        def should_generate_file(rel_path: str) -> bool:
            """Determine if a file should be generated based on component usage."""
            # Always generate non-chart component files
            if 'charts' + os.sep not in rel_path and 'table' + os.sep not in rel_path:
                return True
            
            # Check if this is a chart/table component file
            file_name = os.path.basename(rel_path)
            
            # Check each component type
            for component_type, template_files in component_template_map.items():
                if file_name in template_files:
                    # Only generate if this component type is used
                    return component_type in used_component_types
            
            # Generate other files in charts/table folders (like index files, utilities, etc.)
            return True

        def generate_file_from_template(rel_template_path: str):
            if rel_template_path.endswith(".j2"):
                rel_output_path = rel_template_path[:-3]
            else:
                rel_output_path = rel_template_path

            file_path = self.build_generation_path(file_name=rel_output_path)
            # print(f"Generating file: {file_path} from template: {rel_template_path}")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                template = self.env.get_template(rel_template_path.replace("\\", "/"))
                generated_code = template.render(**context)
                with open(file_path, mode="w", encoding="utf-8") as f:
                    f.write(generated_code)
                # print(f"Code generated in the location: {file_path}")
            except Exception as exc:
                print(f"Error generating {file_path} from {rel_template_path}: {exc}")
                raise

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        for root, _, files in os.walk(templates_path):
            for file_name in files:
                abs_template_path = os.path.join(root, file_name)
                rel_template_path = os.path.relpath(abs_template_path, templates_path)
                
                # Skip files that aren't needed based on component usage
                if not should_generate_file(rel_template_path):
                    # print(f"Skipping unused component: {rel_template_path}")
                    continue
                
                if file_name.endswith(".j2"):
                    # Template files - render with Jinja2
                    generate_file_from_template(rel_template_path)
                else:
                    # Static files (images, etc.) - copy directly
                    rel_output_path = rel_template_path
                    dest_path = self.build_generation_path(file_name=rel_output_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(abs_template_path, dest_path)

    # --------------------------------------------------------------------- #
    # Context builders
    # --------------------------------------------------------------------- #
    def _build_generation_context(self) -> Dict[str, Any]:
        components_payload, styles_payload, meta = self._serialize_gui_model()
        
        # Collect all unique field names from all FieldColumns across all Tables
        all_table_fields = self._collect_all_table_fields()
        
        # Get used component types for conditional rendering in templates
        used_component_types = self._get_used_component_types()
        
        return {
            "model": self.gui_model,
            "components_json": self._to_pretty_json(components_payload),
            "styles_json": self._to_pretty_json(styles_payload),
            "main_page_id": meta.get("main_page_id"),
            "main_page_name": meta.get("main_page_name"),
            "pages_meta": meta.get("pages", []),
            "module_name": meta.get("module_name"),
            "all_table_fields": all_table_fields,
            "used_components": used_component_types,
        }

    def _serialize_gui_model(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        pages_meta: List[Dict[str, str]] = []
        self._style_map = {}

        modules = self._sorted_by_name(self.gui_model.modules)
        module_name = modules[0].name if modules else "AppModule"
        main_page_id: Optional[str] = None
        main_page_name: Optional[str] = None

        for module in modules:
            for screen in self._sorted_by_name(module.screens):
                screen_node = self._serialize_screen(screen)
                pages.append(screen_node)
                pages_meta.append(
                    {
                        "id": screen_node["id"],
                        "name": screen_node.get("name") or screen_node["id"],
                        "route_path": screen_node.get("route_path"),
                    }
                )
                if screen_node.get("is_main") and main_page_id is None:
                    main_page_id = screen_node["id"]
                    main_page_name = screen_node.get("name")

        if main_page_id is None and pages:
            main_page_id = pages[0]["id"]
            main_page_name = pages[0].get("name")

        # Merge raw style entries from the modelling editor LAST
        # GrapesJS styles should COMPLETELY override BUML-generated styles
        for entry in self._raw_style_entries:
            selectors = entry.get("selectors") or []
            grapesjs_style = self._convert_style_keys(entry.get("style") or {})
            
            # MODERNIZE GRAPESJS GRID SYSTEM: Convert table-based grid to flexbox
            if ".gjs-row" in selectors or "gjs-row" in selectors:
                # Convert table display to flex for better responsiveness
                if grapesjs_style.get("display") == "table":
                    grapesjs_style["display"] = "flex"
                    grapesjs_style["flexWrap"] = "wrap"  # Allow wrapping for responsive layout
            
            if ".gjs-cell" in selectors or "gjs-cell" in selectors:
                # Convert table-cell to flex item with proper sizing for 3-column responsive layout
                if grapesjs_style.get("display") in ("table-cell", "block"):
                    # Don't force display for cells - let them be flex items
                    if "display" in grapesjs_style:
                        del grapesjs_style["display"]
                
                # Always use flex-basis for gjs-cell to get proper 3-column layout
                # GrapesJS uses outdated table widths (8%), modernize to flex
                grapesjs_style["flex"] = "1 1 calc(33.333% - 20px)"  # 3 columns with gap
                grapesjs_style["minWidth"] = "250px"  # Min width for responsive wrapping
                # Remove width if present (table-based sizing)
                if "width" in grapesjs_style:
                    del grapesjs_style["width"]
                # Remove height if present (fixed heights break responsiveness)
                if "height" in grapesjs_style:
                    del grapesjs_style["height"]
            
            if selectors:
                # Convert tuple for lookup
                selector_tuple = tuple(selectors)
                # Check if entry already exists from BUML
                existing = self._style_map.get(selector_tuple)
                if existing:
                    # GrapesJS style should override existing BUML style completely
                    # Keep only chart custom properties from BUML, replace everything else
                    buml_chart_props = {k: v for k, v in existing["style"].items() 
                                       if k.startswith("--chart-")}
                    # Start with GrapesJS style, then add back chart props if not overridden
                    merged = dict(grapesjs_style)
                    for chart_key, chart_val in buml_chart_props.items():
                        if chart_key not in merged:
                            merged[chart_key] = chart_val
                    existing["style"] = merged
                else:
                    # New entry - just add it
                    self._add_style_entry(selector_tuple, grapesjs_style)

        styles_payload = {"styles": list(self._style_map.values())}
        components_payload = {"pages": pages}
        meta = {
            "main_page_id": main_page_id,
            "main_page_name": main_page_name,
            "pages": pages_meta,
            "module_name": module_name,
        }
        return components_payload, styles_payload, meta

    # --------------------------------------------------------------------- #
    # Screen & component serialization
    # --------------------------------------------------------------------- #
    def _serialize_screen(self, screen: ViewContainer) -> Dict[str, Any]:
        # Screens may have page_id (preferred) or component_id, fallback to name
        # Note: page_id is set by processor.py for GrapesJS pages
        screen_id = getattr(screen, 'page_id', None) or screen.component_id or screen.name
        screen_name = screen.name
        
        node: Dict[str, Any] = {
            "id": screen_id,
            "name": screen.description or self._humanize(screen_name),
            "description": screen.description or "",
            "is_main": bool(screen.is_main_page),
            "route_path": screen.route_path or f"/{screen_name}".lower().replace(' ', '-'),
            "components": [],
        }

        self._register_component_style(screen_id, screen.styling, getattr(screen, "layout", None))

        # Sort elements by display_order to preserve JSON ordering
        elements = list(screen.view_elements)
        elements.sort(key=lambda e: (e.display_order, e.name))
        
        for element in elements:
            node["components"].append(self._serialize_component(element))

        return self._clean_dict(node)

    def _serialize_component(self, element: ViewComponent) -> Dict[str, Any]:
        component_type = self._map_component_type(element)

        # Use metadata from BUML metamodel (no getattr needed - these are real attributes!)
        # Fallback to defaults only if None
        component_id = element.component_id or element.name or f"{element.__class__.__name__}"
        tag = element.tag_name
        class_list = element.css_classes
        attributes = element.custom_attributes
        display_order = element.display_order

        node: Dict[str, Any] = {
            "id": component_id,
            "type": component_type,
            "name": self._derive_display_name(element),
            "description": element.description or "",
            "tag": tag,
            "class_list": class_list,
            "attributes": attributes,
            "display_order": display_order,
        }

        layout = getattr(element, "layout", None)
        self._register_component_style(component_id, element.styling, layout)

        if isinstance(element, ViewContainer):
            # Sort children by display_order to preserve JSON ordering
            children_list = list(element.view_elements)
            children_list.sort(key=lambda e: (e.display_order, e.name))
            
            children = [self._serialize_component(child) for child in children_list]
            if children:
                node["children"] = children

        if isinstance(element, Text):
            # Ensure content is properly extracted and not empty
            content = element.content
            # If content is empty but we have a tag like h1, h2, h3, p, try to derive from name
            if not content and tag and tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div']:
                # Use name or description as fallback
                content = self._derive_display_name(element) or element.description or ""
            node["content"] = content or ""

        if isinstance(element, Image):
            node["alt"] = element.description or ""
            node["src"] = getattr(element, "source", None)

        if isinstance(element, Link):
            node["label"] = element.label
            node["url"] = element.url
            node["target"] = element.target
            node["rel"] = element.rel

        if isinstance(element, EmbeddedContent):
            node["src"] = element.source
            node["content_type"] = element.content_type or element.name
            node["extra_props"] = element.extra_props or None

        if isinstance(element, Button):
            node["label"] = element.label
            node["button_type"] = self._enum_value(getattr(element, "buttonType", None))
            node["action_type"] = self._enum_value(getattr(element, "actionType", None))
            target_screen = getattr(element, "targetScreen", None)
            if target_screen:
                node["target_screen"] = getattr(target_screen, "name", None)
                node["target_screen_path"] = getattr(target_screen, "route_path", None)
            events = self._serialize_events(element)
            if events:
                node["events"] = events

            # Handle method execution using method_btn (Method object reference)
            method_btn = getattr(element, "method_btn", None)
            if method_btn:
                # Get method name and class from the Method object
                method_name = method_btn.name
                method_class = method_btn.owner
                class_name = method_class.name if method_class else None

                if method_name and class_name:
                    # Clean the method name (remove visibility and parameters if present)
                    clean_method_name = method_name.split('(')[0] if '(' in method_name else method_name

                    # Update attributes with clean names (not IDs)
                    if 'data-method-name' in attributes:
                        attributes['method-name'] = clean_method_name
                        del attributes['data-method-name']

                    if 'data-method-class' in attributes:
                        attributes['method-class'] = class_name
                        del attributes['data-method-class']

                    # Generate endpoint URL for method execution
                    # Format: /book/{book_id}/methods/decrease_stock/
                    class_name_lower = class_name.lower()
                    endpoint = f"/{class_name_lower}/{{{class_name_lower}_id}}/methods/{clean_method_name}/"
                    attributes['endpoint'] = endpoint

                    # Use is_instance_method from Button object
                    is_instance = getattr(element, "is_instance_method", False)
                    attributes['is-instance-method'] = str(is_instance).lower()

                    # Extract input parameters from the method
                    if hasattr(method_btn, 'parameters') and method_btn.parameters:
                        input_params = {}
                        for param in method_btn.parameters:
                            # Skip 'self' and 'session' parameters
                            if param.name.lower() not in ('self', 'session'):
                                param_type = param.type.name if param.type else 'any'
                                input_params[param.name] = param_type
                        
                        if input_params:
                            attributes['input-parameters'] = input_params

            # Output instance_source (table/component ID providing instance data)
            instance_source = getattr(element, "instance_source", None)
            if instance_source:
                # If it's a ViewComponent object, get its component_id; otherwise use the string ID
                if hasattr(instance_source, 'component_id'):
                    attributes['instance-source'] = instance_source.component_id
                elif isinstance(instance_source, str):
                    attributes['instance-source'] = instance_source

        if isinstance(element, InputField):
            node["input_type"] = self._enum_value(getattr(element, "field_type", None))
            node["validation"] = getattr(element, "validationRules", None)

        if isinstance(element, Form):
            inputs = []
            for input_field in sorted(
                getattr(element, "inputFields", []), key=lambda f: getattr(f, "name", "").lower()
            ):
                inputs.append(
                    self._clean_dict(
                        {
                            "id": getattr(input_field, "name", None),
                            "label": self._humanize(getattr(input_field, "name", "")),
                            "type": self._enum_value(getattr(input_field, "field_type", None)),
                            "validation": getattr(input_field, "validationRules", None),
                        }
                    )
                )
            if inputs:
                node["inputs"] = inputs

        if isinstance(element, Menu):
            items = []
            for item in self._sorted_menu_items(element.menuItems):
                items.append(
                    self._clean_dict(
                        {
                            "label": item.label,
                            "url": getattr(item, "url", None),
                            "target": getattr(item, "target", None),
                            "rel": getattr(item, "rel", None),
                        }
                    )
                )
            if items:
                node["items"] = items

        if isinstance(element, DataList):
            sources = []
            for source in self._sorted_by_name(element.list_sources):
                domain_name = getattr(getattr(source, "dataSourceClass", None), "name", None)
                field_names = [
                    getattr(field, "name", None)
                    for field in getattr(source, "fields", [])
                    if getattr(field, "name", None)
                ]
                if not field_names and getattr(source, "field_names", None):
                    field_names = list(getattr(source, "field_names"))
                fields = sorted(field_names) if field_names else None
                label_field_name = getattr(getattr(source, "label_field", None), "name", None) or getattr(
                    source, "label_field_name", None
                )
                value_field_name = getattr(getattr(source, "value_field", None), "name", None) or getattr(
                    source, "value_field_name", None
                )
                sources.append(
                    self._clean_dict(
                        {
                            "name": source.name,
                            "domain": domain_name,
                            "fields": fields,
                            "label_field": label_field_name,
                            "value_field": value_field_name,
                        }
                    )
                )
            if sources:
                node["data_sources"] = sources

        if isinstance(element, LineChart):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "lineWidth": element.line_width,
                    "curveType": element.curve_type,
                    "showGrid": element.show_grid,
                    "showLegend": element.show_legend,
                    "showTooltip": element.show_tooltip,
                    "animate": element.animate,
                    "legendPosition": element.legend_position,
                    "gridColor": element.grid_color,
                    "dotSize": element.dot_size,
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("line") or "#4CAF50"

        if isinstance(element, BarChart):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "barWidth": element.bar_width,
                    "orientation": element.orientation,
                    "showGrid": element.show_grid,
                    "showLegend": element.show_legend,
                    "showTooltip": element.show_tooltip,
                    "stacked": element.stacked,
                    "animate": element.animate,
                    "legendPosition": element.legend_position,
                    "gridColor": element.grid_color,
                    "barGap": element.bar_gap,
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("bar") or "#3498db"

        if isinstance(element, PieChart):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "showLegend": element.show_legend,
                    "legendPosition": self._enum_value(element.legend_position),
                    "showLabels": element.show_labels,
                    "labelPosition": self._enum_value(element.label_position),
                    "paddingAngle": element.padding_angle,
                    "innerRadius": element.inner_radius,
                    "outerRadius": element.outer_radius,
                    "startAngle": element.start_angle,
                    "endAngle": element.end_angle,
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("palette") or "#8884d8"

        if isinstance(element, RadarChart):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "showGrid": element.show_grid,
                    "showTooltip": element.show_tooltip,
                    "showLegend": element.show_legend,
                    "legendPosition": element.legend_position,
                    "dotSize": element.dot_size,
                    "gridType": element.grid_type,
                    "strokeWidth": element.stroke_width,
                    "showRadiusAxis": element.show_radius_axis,
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("palette") or "#8884d8"

        if isinstance(element, RadialBarChart):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "startAngle": element.start_angle,
                    "endAngle": element.end_angle,
                    "innerRadius": element.inner_radius,
                    "outerRadius": element.outer_radius,
                    "showLegend": element.show_legend,
                    "legendPosition": element.legend_position,
                    "showTooltip": element.show_tooltip,
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("palette") or "#8884d8"

        if isinstance(element, Table):
            node["title"] = element.title or self._humanize(element.name)
            node["chart"] = self._clean_dict(
                {
                    "showHeader": element.show_header,
                    "stripedRows": element.striped_rows,
                    "showPagination": element.show_pagination,
                    "rowsPerPage": element.rows_per_page,
                    "actionButtons": getattr(element, "action_buttons", False),
                }
            )
            chart_colors = self._extract_chart_colors(element)
            node["color"] = element.primary_color or chart_colors.get("background") or "#2c3e50"

            # Serialize columns from Column objects
            columns = []
            for col in getattr(element, "columns", []) or []:
                column_dict = {"label": col.label}

                if isinstance(col, FieldColumn):
                    column_dict["column_type"] = "field"
                    column_dict["field"] = col.field.name if hasattr(col.field, "name") else str(col.field)

                    # Check if the field type is an Enumeration
                    field_type = getattr(col.field, "type", None)
                    if isinstance(field_type, Enumeration):
                        column_dict["type"] = "enum"
                        # Extract enumeration literal names as options
                        column_dict["options"] = sorted([literal.name for literal in field_type.literals])
                    else:
                        column_dict["type"] = field_type.name if hasattr(field_type, "name") else "str"
                    
                    # Check if the field is required based on minimum multiplicity
                    if hasattr(col.field, "multiplicity") and hasattr(col.field.multiplicity, "min"):
                        column_dict["required"] = col.field.multiplicity.min > 0
                    else:
                        column_dict["required"] = False

                elif isinstance(col, LookupColumn):
                    column_dict["column_type"] = "lookup"
                    column_dict["path"] = col.path.name if hasattr(col.path, "name") else str(col.path)
                    column_dict["entity"] = col.path.type.name if hasattr(col.path, "type") and hasattr(col.path.type, "name") else ""
                    column_dict["field"] = col.field.name if hasattr(col.field, "name") else str(col.field)
                    # Determine type based on path multiplicity
                    if hasattr(col.path, "multiplicity") and hasattr(col.path.multiplicity, "max"):
                        is_list = col.path.multiplicity.max > 1 or col.path.multiplicity.max == "*"
                        if is_list:
                            column_dict["type"] = "list"
                        else:
                            column_dict["type"] = getattr(col.field, "type", {}).name if hasattr(getattr(col.field, "type", None), "name") else "str"
                    else:
                        column_dict["type"] = getattr(col.field, "type", {}).name if hasattr(getattr(col.field, "type", None), "name") else "str"
                    
                    # Check if the lookup path is required based on minimum multiplicity
                    if hasattr(col.path, "multiplicity") and hasattr(col.path.multiplicity, "min"):
                        column_dict["required"] = col.path.multiplicity.min > 0
                    else:
                        column_dict["required"] = False

                elif isinstance(col, ExpressionColumn):
                    column_dict["column_type"] = "expression"
                    column_dict["field"] = col.label
                    column_dict["expression"] = col.expression
                    column_dict["type"] = "computed"

                columns.append(column_dict)

            if columns:
                node["chart"]["columns"] = columns
            
            # Remove raw GrapesJS columns from attributes - we use the processed columns in chart
            if "attributes" in node and isinstance(node["attributes"], dict) and "columns" in node["attributes"]:
                node["attributes"] = {k: v for k, v in node["attributes"].items() if k != "columns"}

        if isinstance(element, MetricCard):
            node["title"] = element.metric_title or self._humanize(element.name)
            node["metric"] = self._clean_dict(
                {
                    "metricTitle": element.metric_title,
                    "format": element.format,
                    "valueColor": element.value_color,
                    "valueSize": element.value_size,
                    "showTrend": element.show_trend,
                    "positiveColor": element.positive_color,
                    "negativeColor": element.negative_color,
                    "value": 0,  # Default placeholder - will be populated from API
                    "trend": 12,  # Default placeholder trend percentage
                }
            )
            node["color"] = element.primary_color or element.value_color or "#2c3e50"

        if isinstance(element, AgentComponent):
            node["agent-name"] = element.agent_name or ""
            node["agent-title"] = element.agent_title or "BESSER Agent"
            node["agent_name"] = element.agent_name or ""
            node["agent_title"] = element.agent_title or "BESSER Agent"

        binding_data = self._serialize_data_binding(getattr(element, "data_binding", None))
        if binding_data:
            node["data_binding"] = binding_data

        return self._clean_dict(node)

    def _serialize_events(self, element: Button) -> Optional[List[Dict[str, Any]]]:
        events = getattr(element, "events", None)
        if not events:
            return None

        serialized: List[Dict[str, Any]] = []
        for event in self._sorted_by_name(events):
            actions = [
                action_data
                for action in self._sorted_by_name(event.actions)
                if (action_data := self._serialize_action(action))
            ]
            event_dict = self._clean_dict(
                {
                    "name": event.name,
                    "type": self._enum_value(getattr(event, "event_type", None)),
                    "actions": actions,
                }
            )
            if event_dict.get("actions"):
                serialized.append(event_dict)
        return serialized or None

    def _serialize_action(self, action) -> Optional[Dict[str, Any]]:
        if action is None:
            return None

        data: Dict[str, Any] = {
            "name": getattr(action, "name", None),
            "kind": action.__class__.__name__,
            "description": getattr(action, "description", None),
        }

        if isinstance(action, Transition):
            target_screen = getattr(action, "target_screen", None)
            data["target_screen"] = getattr(target_screen, "name", None)
            data["target_screen_id"] = getattr(action, "_target_screen_id", None)
            data["target_screen_path"] = getattr(target_screen, "route_path", None)

        if isinstance(action, (Create, Read, Update, Delete)):
            data["target_class"] = getattr(getattr(action, "target_class", None), "name", None)

        parameters = getattr(action, "parameters", None)
        if parameters:
            params_serialized = []
            for param in sorted(parameters, key=lambda p: getattr(p, "name", "")):
                params_serialized.append(
                    self._clean_dict(
                        {
                            "name": getattr(param, "name", None),
                            "type": getattr(param, "param_type", None),
                            "required": getattr(param, "required", None),
                            "value": getattr(param, "value", None),
                        }
                    )
                )
            if params_serialized:
                data["parameters"] = params_serialized

        return self._clean_dict(data)

    def _serialize_data_binding(self, binding) -> Optional[Dict[str, Any]]:
        if not binding:
            return None

        domain = getattr(binding, "domain_concept", None)
        label_field = getattr(binding, "label_field", None)
        data_field = getattr(binding, "data_field", None)
        data_filter = getattr(binding, "data_filter", None)
        
        endpoint = None
        if domain and getattr(domain, "name", None):
            endpoint = f"/{domain.name.lower()}/"

        return self._clean_dict(
            {
                "entity": getattr(domain, "name", None),
                "endpoint": endpoint,
                "label_field": getattr(label_field, "name", None),
                "data_field": getattr(data_field, "name", None),
                "filter": str(data_filter) if data_filter else None,
            }
        )

    # --------------------------------------------------------------------- #
    # Styling helpers
    # --------------------------------------------------------------------- #
    def _register_component_style(self, selector_id: str, styling, layout=None):
        style = self._style_from_styling(styling, layout)
        # Only register if there are meaningful styles (not just defaults)
        if style and self._has_meaningful_styles(style):
            self._add_style_entry((f"#{selector_id}",), style)

    def _style_from_styling(self, styling, layout=None) -> Dict[str, Any]:
        if not styling:
            return {}

        style: Dict[str, Any] = {}
        size = getattr(styling, "size", None)
        position = getattr(styling, "position", None)
        color = getattr(styling, "color", None)

        if size:
            style.update(self._extract_size_style(size))
        if position:
            style.update(self._extract_position_style(position))
        if color:
            style.update(self._extract_color_style(color))

        if layout is None and hasattr(styling, "layout"):
            layout = styling.layout
        if layout:
            style.update(self._extract_layout_style(layout))

        return self._filter_style(style)

    def _extract_size_style(self, size) -> Dict[str, Any]:
        style: Dict[str, Any] = {}
        if getattr(size, "width", None):
            style["width"] = size.width
        if getattr(size, "height", None):
            style["height"] = size.height
        if getattr(size, "padding", None):
            style["padding"] = size.padding
        if getattr(size, "margin", None):
            style["margin"] = size.margin
        if getattr(size, "font_size", None):
            style["fontSize"] = size.font_size
        if getattr(size, "line_height", None):
            style["lineHeight"] = size.line_height
        if getattr(size, "icon_size", None) and "fontSize" not in style:
            style["fontSize"] = size.icon_size
        return style

    def _extract_position_style(self, position) -> Dict[str, Any]:
        style: Dict[str, Any] = {}
        position_type = getattr(position, "p_type", None)
        if position_type:
            style["position"] = self._enum_value(position_type)

        for attr in ("top", "left", "right", "bottom"):
            value = getattr(position, attr, None)
            if value not in (None, "auto"):
                style[attr] = value

        alignment = getattr(position, "alignment", None)
        if alignment:
            style["textAlign"] = self._enum_value(alignment) or alignment

        if getattr(position, "z_index", None) is not None:
            style["zIndex"] = position.z_index

        return style

    def _extract_color_style(self, color) -> Dict[str, Any]:
        style: Dict[str, Any] = {}
        if getattr(color, "background_color", None):
            bg_color = str(color.background_color).replace(" !important", "").replace("!important", "").strip()
            if "linear-gradient" in bg_color or "radial-gradient" in bg_color:
                style["backgroundImage"] = bg_color
            else:
                style["backgroundColor"] = bg_color
        if getattr(color, "text_color", None):
            style["color"] = str(color.text_color).replace(" !important", "").replace("!important", "").strip()
        if getattr(color, "border_color", None):
            style["borderColor"] = str(color.border_color).replace(" !important", "").replace("!important", "").strip()
        if getattr(color, "opacity", None) not in (None, ""):
            style["opacity"] = color.opacity
        if getattr(color, "box_shadow", None):
            style["boxShadow"] = color.box_shadow
        if getattr(color, "gradient", None):
            style["backgroundImage"] = color.gradient
        # Preserve chart colors as CSS custom properties for styling fallbacks
        if getattr(color, "line_color", None):
            style["--chart-line-color"] = color.line_color
        if getattr(color, "bar_color", None):
            style["--chart-bar-color"] = color.bar_color
        if getattr(color, "color_palette", None):
            style["--chart-color-palette"] = color.color_palette
        if getattr(color, "radius", None):
            style["borderRadius"] = color.radius
        return style

    def _extract_layout_style(self, layout) -> Dict[str, Any]:
        style: Dict[str, Any] = {}

        layout_type = self._enum_value(getattr(layout, "layout_type", None))
        orientation = getattr(layout, "orientation", None)

        if layout_type in {"flex", "row", "column"}:
            style["display"] = "flex"
        elif layout_type == "grid":
            style["display"] = "grid"
        elif layout_type == "absolute":
            style["position"] = "absolute"

        flex_direction = getattr(layout, "flex_direction", None)
        if flex_direction:
            # Only add flexDirection if explicitly set in GrapesJS
            style["flexDirection"] = flex_direction
        elif orientation == "horizontal":
            style["flexDirection"] = "row"
        elif orientation == "vertical":
            style["flexDirection"] = "column"
        elif layout_type == "row":
            style["flexDirection"] = "row"
        elif layout_type == "column":
            style["flexDirection"] = "column"
        # DON'T add default flexDirection - let CSS handle it (default is row anyway)

        if getattr(layout, "justify_content", None):
            style["justifyContent"] = layout.justify_content
        elif getattr(layout, "alignment", None):
            alignment = self._enum_value(layout.alignment)
            if alignment:
                style.setdefault("justifyContent", alignment)

        if getattr(layout, "align_items", None):
            style["alignItems"] = layout.align_items

        if getattr(layout, "flex_wrap", None):
            style["flexWrap"] = layout.flex_wrap
        elif getattr(layout, "wrap", None) is not None:
            style["flexWrap"] = "wrap" if layout.wrap else "nowrap"

        if getattr(layout, "gap", None):
            style["gap"] = layout.gap
        if getattr(layout, "grid_template_columns", None):
            style["gridTemplateColumns"] = layout.grid_template_columns
        if getattr(layout, "grid_template_rows", None):
            style["gridTemplateRows"] = layout.grid_template_rows
        if getattr(layout, "grid_gap", None):
            style["gridGap"] = layout.grid_gap

        if getattr(layout, "padding", None):
            style.setdefault("padding", layout.padding)
        if getattr(layout, "margin", None):
            style.setdefault("margin", layout.margin)

        return style

    def _extract_chart_colors(self, element: ViewComponent) -> Dict[str, Any]:
        color = getattr(getattr(element, "styling", None), "color", None)
        if not color:
            return {}
        return {
            "line": getattr(color, "line_color", None),
            "bar": getattr(color, "bar_color", None),
            "palette": getattr(color, "color_palette", None),
            "background": getattr(color, "background_color", None),
        }

    def _add_style_entry(self, selectors: Sequence[str], style: Dict[str, Any]):
        normalized_selectors = tuple(sorted({selector for selector in selectors if selector}))
        if not normalized_selectors or not style:
            return
        entry = self._style_map.get(normalized_selectors)
        if entry:
            entry["style"].update(style)
        else:
            self._style_map[normalized_selectors] = {
                "selectors": list(normalized_selectors),
                "style": dict(style),
            }

    def _has_meaningful_styles(self, style: Dict[str, Any]) -> bool:
        """Check if the style dict has non-default meaningful values."""
        # These are default values that don't need to be exported
        defaults = {
            "backgroundColor": "#FFFFFF",
            "color": "#000000",
            "borderColor": "#CCCCCC",
            "margin": "0",
            "padding": "0",
            "height": "auto",
            "width": "auto",
            "position": "static",
            "textAlign": "left",
            "zIndex": 0,
            "--chart-line-color": "#000000",
            "--chart-bar-color": "#CCCCCC",
            "--chart-color-palette": "default",
        }
        
        # Check if there's at least one non-default value
        for key, value in style.items():
            if key not in defaults:
                # This is a non-default property
                return True
            if defaults.get(key) != value:
                # This is a default property but with a different value
                return True
        
        # All values are defaults
        return False

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_pretty_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _sorted_by_name(items: Iterable[Any]) -> List[Any]:
        if not items:
            return []
        # Sort by display_order if available (preserves JSON order), otherwise by name
        return sorted(items, key=lambda item: (getattr(item, "display_order", 999999), getattr(item, "name", "").lower()))

    @staticmethod
    def _sorted_menu_items(items: Iterable[MenuItem]) -> List[MenuItem]:
        if not items:
            return []
        return sorted(items, key=lambda item: (item.label or "").lower())

    @staticmethod
    def _sanitize_class_list(class_list) -> Optional[List[str]]:
        if not class_list:
            return None
        names: List[str] = []
        for item in class_list:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(str(name))
            elif isinstance(item, str):
                names.append(item)
        return names or None

    @staticmethod
    def _sanitize_attributes(attributes) -> Dict[str, Any]:
        if not isinstance(attributes, dict):
            return {}
        safe: Dict[str, Any] = {}
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            else:
                safe[key] = str(value)
        return safe

    @staticmethod
    def _map_component_type(element: ViewComponent) -> str:
        if isinstance(element, ViewContainer):
            return "container"
        if isinstance(element, Text):
            return "text"
        if isinstance(element, Image):
            return "image"
        if isinstance(element, Link):
            return "link"
        if isinstance(element, Button):
            return "button"
        if isinstance(element, InputField):
            return "input"
        if isinstance(element, Form):
            return "form"
        if isinstance(element, EmbeddedContent):
            return "embedded-content"
        if isinstance(element, Menu):
            return "menu"
        if isinstance(element, DataList):
            return "data-list"
        if isinstance(element, LineChart):
            return "line-chart"
        if isinstance(element, BarChart):
            return "bar-chart"
        if isinstance(element, PieChart):
            return "pie-chart"
        if isinstance(element, RadarChart):
            return "radar-chart"
        if isinstance(element, RadialBarChart):
            return "radial-bar-chart"
        if isinstance(element, Table):
            return "table"
        if isinstance(element, MetricCard):
            return "metric-card"
        if isinstance(element, AgentComponent):
            return "agent-component"
        return "component"

    @staticmethod
    def _derive_display_name(element: ViewComponent) -> str:
        title = getattr(element, "title", None)
        if isinstance(title, str) and title.strip():
            return title.strip()
        if isinstance(element, Text):
            content = getattr(element, "content", "")
            if isinstance(content, str):
                trimmed = content.strip()
                if trimmed:
                    return trimmed
        return ReactGenerator._humanize(element.name)

    @staticmethod
    def _humanize(value: Optional[str]) -> str:
        if not value:
            return ""
        return value.replace("_", " ").strip()

    def _collect_all_table_fields(self) -> List[str]:
        """
        Collect all unique field names from FieldColumns across all Tables in the GUI model.
        Recursively searches through all ViewContainers to find nested Tables.
        These will be used to generate dynamic option display logic in the template.
        
        Returns:
            List of unique field names sorted alphabetically
        """
        field_names = set()
        
        def scan_element_for_tables(element):
            """Recursively scan an element and its children for Tables."""
            if isinstance(element, Table):
                # Extract field names from FieldColumns
                for col in getattr(element, "columns", []) or []:
                    if isinstance(col, FieldColumn):
                        field_name = col.field.name if hasattr(col.field, "name") else str(col.field)
                        field_names.add(field_name)
            
            # Recursively scan children if this is a ViewContainer
            if isinstance(element, ViewContainer):
                for child in element.view_elements:
                    scan_element_for_tables(child)
        
        # Iterate through all modules and screens to find Tables (including nested ones)
        for module in self.gui_model.modules:
            for screen in module.screens:
                for element in screen.view_elements:
                    scan_element_for_tables(element)
        
        return sorted(field_names)

    def _get_used_component_types(self) -> set:
        """
        Scan the GUI model to determine which component types are actually used.
        Recursively searches through all ViewContainers to find nested components.
        
        Returns:
            Set of component type names (e.g., 'LineChart', 'BarChart', 'Table', etc.)
        """
        used_types = set()
        
        def scan_element(element):
            """Recursively scan an element and its children for component types."""
            # Map component instances to their type names
            if isinstance(element, LineChart):
                used_types.add('LineChart')
            elif isinstance(element, BarChart):
                used_types.add('BarChart')
            elif isinstance(element, PieChart):
                used_types.add('PieChart')
            elif isinstance(element, RadarChart):
                used_types.add('RadarChart')
            elif isinstance(element, RadialBarChart):
                used_types.add('RadialBarChart')
            elif isinstance(element, Table):
                used_types.add('Table')
            elif isinstance(element, MetricCard):
                used_types.add('MetricCard')
            
            # Recursively scan children if this is a ViewContainer
            if isinstance(element, ViewContainer):
                for child in element.view_elements:
                    scan_element(child)
        
        # Scan all screens and their children
        for module in self.gui_model.modules:
            for screen in module.screens:
                for element in screen.view_elements:
                    scan_element(element)
        
        return used_types

    @staticmethod
    def _enum_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Enum):
            return value.value if hasattr(value, "value") else value.name
        return str(value)
    
    def _get_entity_name_from_id(self, entity_id: str) -> Optional[str]:
        """Get entity name from the structural model by ID"""
        if not hasattr(self, 'model') or not self.model:
            return None
        
        # Search for class with matching name or component_id
        for cls in self.model.get_classes():
            if cls.name == entity_id or getattr(cls, 'component_id', None) == entity_id:
                return cls.name
        
        # If not found, return the ID as-is (it might already be a name)
        return entity_id
    
    @staticmethod
    def _infer_type(value: Any) -> str:
        """Infer parameter type from value"""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'number'
        elif isinstance(value, float):
            return 'number'
        elif isinstance(value, str):
            return 'string'
        else:
            return 'string'

    @staticmethod
    def _convert_style_keys(style: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for key, value in style.items():
            if value in (None, ""):
                continue
            # Strip !important flags as React inline styles don't support them
            clean_value = str(value).replace(" !important", "").replace("!important", "").strip() if isinstance(value, str) else value
            # Handle gradients - they need to go in backgroundImage, not backgroundColor
            camel_key = ReactGenerator._to_camel_case(key)
            if camel_key == "backgroundImage" and isinstance(clean_value, str):
                # Already correct property for gradients
                converted[camel_key] = clean_value
            elif camel_key == "backgroundColor" and isinstance(clean_value, str) and ("linear-gradient" in clean_value or "radial-gradient" in clean_value):
                # Move gradients from backgroundColor to backgroundImage
                converted["backgroundImage"] = clean_value
            else:
                converted[camel_key] = clean_value
        return converted

    @staticmethod
    def _to_camel_case(name: str) -> str:
        if not name:
            return name
        parts = name.replace("-", "_").split("_")
        first, *rest = parts
        camel = first
        for part in rest:
            if part:
                camel += part[0].upper() + part[1:]
        return camel

    @staticmethod
    def _filter_style(style: Dict[str, Any]) -> Dict[str, Any]:
        filtered: Dict[str, Any] = {}
        for key, value in style.items():
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            filtered[key] = value
        return filtered

    @staticmethod
    def _clean_dict(data: Dict[str, Any], preserve_keys: set = None) -> Dict[str, Any]:
        """Clean dict by removing empty/None values.
        
        Args:
            data: Dictionary to clean
            preserve_keys: Set of keys to preserve even if their value is empty
        """
        if preserve_keys is None:
            preserve_keys = {"components", "children"}  # Always preserve these for TypeScript compatibility
        
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, bool):
                cleaned[key] = value
                continue
            # Preserve specified keys even if empty (for TypeScript type safety)
            if key in preserve_keys:
                cleaned[key] = value if value is not None else []
                continue
            if value in (None, "", [], {}, ()):
                continue
            cleaned[key] = value
        return cleaned
