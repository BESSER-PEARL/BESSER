from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set


class _PageRenderContext:
    def __init__(self) -> None:
        self.imports: Set[str] = set()
        self.needs_navigate: bool = False


class PageBuilderMixin:
    def _generate_pages(self):
        """Build real TSX pages and App.tsx, and remove legacy renderer artifacts."""
        components_payload, _, meta = self._serialize_gui_model()
        pages = components_payload.get("pages", [])
        self._page_path_map = self._build_page_path_map(pages)
        selector_style_map = self._build_selector_style_map()

        page_infos = self._write_page_components(pages, selector_style_map)
        self._write_app_tsx(page_infos, meta)

    def _build_selector_style_map(self) -> Dict[str, Dict[str, Any]]:
        selector_style_map: Dict[str, Dict[str, Any]] = {}
        for entry in self._style_map.values():
            for selector in entry.get("selectors", []) or []:
                if selector not in selector_style_map:
                    selector_style_map[selector] = entry.get("style", {}) or {}
        return selector_style_map

    def _write_page_components(
        self, pages: List[Dict[str, Any]], selector_style_map: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pages_dir = self.build_generation_path("src/pages")
        os.makedirs(pages_dir, exist_ok=True)

        used_names: Set[str] = set()
        page_infos: List[Dict[str, Any]] = []

        for idx, page in enumerate(pages):
            raw_name = page.get("name") or page.get("id") or f"Page{idx + 1}"
            component_name = self._to_pascal_case(raw_name)
            if not component_name:
                component_name = f"Page{idx + 1}"
            if component_name[0].isdigit():
                component_name = f"Page{component_name}"

            base_name = component_name
            counter = 1
            while component_name in used_names:
                counter += 1
                component_name = f"{base_name}{counter}"
            used_names.add(component_name)

            context = _PageRenderContext()
            jsx = self._render_page_jsx(page, selector_style_map, context)

            imports = self._build_page_imports(context)
            navigate_hook = "  const navigate = useNavigate();\n" if context.needs_navigate else ""

            file_contents = (
                "import React from \"react\";\n"
                + ("import { useNavigate } from \"react-router-dom\";\n" if context.needs_navigate else "")
                + imports
                + "\n"
                + f"const {component_name}: React.FC = () => {{\n"
                + navigate_hook
                + "  return (\n"
                + jsx
                + "  );\n"
                + "};\n\n"
                + f"export default {component_name};\n"
            )

            file_path = os.path.join(pages_dir, f"{component_name}.tsx")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_contents)

            page_infos.append(
                {
                    "component_name": component_name,
                    "file_name": f"./pages/{component_name}",
                    "route_path": page.get("route_path") or "/",
                    "is_main": bool(page.get("is_main")),
                    "page_id": page.get("id"),
                }
            )

        return page_infos

    @staticmethod
    def _build_page_path_map(pages: List[Dict[str, Any]]) -> Dict[str, str]:
        page_path_map: Dict[str, str] = {}
        for page in pages or []:
            page_id = page.get("id")
            route_path = page.get("route_path")
            if not route_path:
                name = page.get("name") or page_id
                route_path = PageBuilderMixin._path_from_name(name)
            if not route_path:
                continue
            if page_id:
                page_path_map[page_id] = route_path
                page_path_map[f"page:{page_id}"] = route_path
            if page.get("name"):
                page_path_map[str(page["name"])] = route_path
        return page_path_map

    def _build_page_imports(self, context: _PageRenderContext) -> str:
        imports: List[str] = []
        if "ChartBlock" in context.imports:
            imports.append("import { ChartBlock } from \"../components/runtime/ChartBlock\";")
        if "TableBlock" in context.imports:
            imports.append("import { TableBlock } from \"../components/runtime/TableBlock\";")
        if "MetricCardBlock" in context.imports:
            imports.append("import { MetricCardBlock } from \"../components/runtime/MetricCardBlock\";")
        if "DataListBlock" in context.imports:
            imports.append("import { DataListBlock } from \"../components/runtime/DataListBlock\";")
        if "AgentComponent" in context.imports:
            imports.append("import { AgentComponent } from \"../components/AgentComponent\";")
        if "MethodButton" in context.imports:
            imports.append("import { MethodButton } from \"../components/MethodButton\";")

        if not imports:
            return ""
        return "\n".join(imports) + "\n"

    def _render_page_jsx(
        self,
        page: Dict[str, Any],
        selector_style_map: Dict[str, Dict[str, Any]],
        context: _PageRenderContext,
    ) -> str:
        page_id = page.get("id") or ""
        components = page.get("components") or []

        page_style = self._compute_style(
            component_id=page_id,
            class_list=[],
            selector_style_map=selector_style_map,
            has_children=bool(components),
        )

        props = self._build_element_props(
            component_id=page_id,
            class_list=[],
            style=page_style,
            attributes={},
            on_click_expr=None,
        )

        children_jsx = self._render_component_list(components, selector_style_map, context, indent=4)
        if children_jsx:
            return (
                f"    <div{props}>\n"
                + children_jsx
                + "    </div>\n"
            )
        return f"    <div{props} />\n"

    def _render_component_list(
        self,
        components: List[Dict[str, Any]],
        selector_style_map: Dict[str, Dict[str, Any]],
        context: _PageRenderContext,
        indent: int,
    ) -> str:
        rendered = []
        for component in components:
            rendered.append(
                self._render_component(component, selector_style_map, context, indent)
            )
        return "\n".join([line for line in rendered if line])

    def _render_component(
        self,
        node: Dict[str, Any],
        selector_style_map: Dict[str, Dict[str, Any]],
        context: _PageRenderContext,
        indent: int,
    ) -> str:
        comp_type = node.get("type") or "component"
        component_id = node.get("id") or ""
        class_list = node.get("class_list") or []
        attributes = dict(node.get("attributes") or {})
        children = node.get("children") or []

        style = self._compute_style(
            component_id=component_id,
            class_list=class_list,
            selector_style_map=selector_style_map,
            has_children=bool(children),
        )

        indent_str = " " * indent

        # Containers
        if comp_type in {"container", "wrapper", "component"}:
            tag = self._select_container_tag(node.get("tag"))
            props = self._build_element_props(component_id, class_list, style, attributes, None)
            if children:
                children_jsx = self._render_component_list(children, selector_style_map, context, indent + 2)
                return (
                    f"{indent_str}<{tag}{props}>\n"
                    + children_jsx
                    + f"\n{indent_str}</{tag}>"
                )
            return f"{indent_str}<{tag}{props} />"

        # Text
        if comp_type == "text":
            tag = self._select_text_tag(node.get("tag"))
            content = node.get("content") or ""
            content_expr = f"{{{json.dumps(content, ensure_ascii=False)}}}"
            props = self._build_element_props(component_id, class_list, style, attributes, None)
            return f"{indent_str}<{tag}{props}>{content_expr}</{tag}>"

        # Button
        if comp_type in {"button", "action-button"}:
            return self._render_button(node, component_id, class_list, style, attributes, context, indent_str)

        # Link
        if comp_type in {"link", "link-button"}:
            url = node.get("url") or ""
            target_path = self._resolve_target_path(node)
            on_click_expr = None
            if target_path and (not url or url == "#" or not str(url).startswith("http")):
                context.needs_navigate = True
                on_click_expr = (
                    f"(e) => {{ e.preventDefault(); navigate({json.dumps(target_path)}); }}"
                )
                url = target_path
            if not url or url == "#":
                url = target_path or "/"

            link_attrs = dict(attributes)
            link_attrs.pop("href", None)
            props = self._build_element_props(
                component_id,
                class_list,
                style,
                link_attrs,
                on_click_expr,
                extra_props={
                    "href": url,
                    "target": node.get("target") or None,
                    "rel": node.get("rel") or None,
                },
            )
            label = node.get("label") or node.get("name") or ""
            content_expr = f"{{{json.dumps(label, ensure_ascii=False)}}}"
            return f"{indent_str}<a{props}>{content_expr}</a>"

        # Image
        if comp_type == "image":
            img_attrs = dict(attributes)
            img_attrs.pop("src", None)
            img_attrs.pop("alt", None)
            props = self._build_element_props(
                component_id,
                class_list,
                style,
                img_attrs,
                None,
                extra_props={
                    "src": node.get("src") or "",
                    "alt": node.get("alt") or node.get("description") or "",
                },
            )
            return f"{indent_str}<img{props} />"

        # Input
        if comp_type == "input":
            input_type = (node.get("input_type") or attributes.get("type") or "text")
            input_attrs = dict(attributes)
            input_attrs.pop("type", None)
            props = self._build_element_props(
                component_id,
                class_list,
                style,
                input_attrs,
                None,
                extra_props={
                    "type": str(input_type).lower(),
                    "placeholder": attributes.get("placeholder") or "",
                },
            )
            return f"{indent_str}<input{props} />"

        # Form
        if comp_type == "form":
            on_submit_expr = "(e) => { e.preventDefault(); }"
            props = self._build_element_props(component_id, class_list, style, attributes, None)
            props += " onSubmit={" + on_submit_expr + "}"
            inputs = node.get("inputs") or []
            input_lines = []
            for input_field in inputs:
                input_id = input_field.get("id") or ""
                label = input_field.get("label") or input_id
                field_type = (input_field.get("type") or "text").lower()
                label_expr = f"{{{json.dumps(label, ensure_ascii=False)}}}"
                input_lines.append(
                    f"{indent_str}  <div style={{\"marginBottom\": \"10px\"}}>\n"
                    f"{indent_str}    <label htmlFor={json.dumps(input_id)} style={{\"display\": \"block\", \"marginBottom\": \"5px\"}}>"
                    f"{label_expr}"
                    f"</label>\n"
                    f"{indent_str}    <input id={json.dumps(input_id)} type={json.dumps(field_type)} />\n"
                    f"{indent_str}  </div>"
                )
            inner = "\n".join(input_lines) if input_lines else ""
            return (
                f"{indent_str}<form{props}>\n"
                + inner
                + f"\n{indent_str}  <button type=\"submit\">Submit</button>\n"
                + f"{indent_str}</form>"
            )

        # Menu
        if comp_type == "menu":
            items = node.get("items") or []
            props = self._build_element_props(component_id, class_list, style, attributes, None)
            menu_list_style = self._format_prop(
                "style",
                {"listStyle": "none", "padding": 0, "margin": 0},
            )
            list_item_style = self._format_prop(
                "style",
                {"display": "inline-block", "marginRight": "15px"},
            )
            item_lines = []
            for item in items:
                label = item.get("label") or ""
                url = item.get("url") or ""
                if not url or url == "#":
                    url = self._page_path_map.get(label) or "/"
                target = item.get("target")
                rel = item.get("rel")
                label_expr = f"{{{json.dumps(label, ensure_ascii=False)}}}"
                item_lines.append(
                    f"{indent_str}    <li {list_item_style}>"
                    f"<a href={json.dumps(url)}"
                    + (f" target={json.dumps(target)}" if target else "")
                    + (f" rel={json.dumps(rel)}" if rel else "")
                    + f">{label_expr}</a></li>"
                )
            items_jsx = "\n".join(item_lines)
            return (
                f"{indent_str}<nav{props}>\n"
                f"{indent_str}  <ul {menu_list_style}>\n"
                + items_jsx
                + f"\n{indent_str}  </ul>\n"
                + f"{indent_str}</nav>"
            )

        # Data List
        if comp_type == "data-list":
            context.imports.add("DataListBlock")
            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "dataSources": node.get("data_sources") or [],
                },
            )
            return f"{indent_str}<DataListBlock{props} />"

        # Embedded content (iframe)
        if comp_type == "embedded-content":
            iframe_attrs = dict(attributes)
            iframe_attrs.pop("src", None)
            props = self._build_element_props(
                component_id,
                class_list,
                style,
                iframe_attrs,
                None,
                extra_props={
                    "src": node.get("src") or "",
                    "title": node.get("description") or node.get("name") or "Embedded content",
                },
            )
            return f"{indent_str}<iframe{props} />"

        # Charts
        if comp_type in {"bar-chart", "line-chart", "pie-chart", "radar-chart", "radial-bar-chart"}:
            context.imports.add("ChartBlock")
            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "chartType": comp_type,
                    "title": node.get("title"),
                    "color": node.get("color"),
                    "chart": node.get("chart"),
                    "series": node.get("series"),
                    "dataBinding": node.get("data_binding"),
                },
                style_prop_name="styles",
                include_class_name=False,
            )
            return f"{indent_str}<ChartBlock{props} />"

        # Table
        if comp_type == "table":
            context.imports.add("TableBlock")
            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "title": node.get("title"),
                    "options": node.get("chart"),
                    "dataBinding": node.get("data_binding"),
                },
                style_prop_name="styles",
                include_class_name=False,
            )
            return f"{indent_str}<TableBlock{props} />"

        # Metric card
        if comp_type == "metric-card":
            context.imports.add("MetricCardBlock")
            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "metric": node.get("metric"),
                    "dataBinding": node.get("data_binding"),
                },
                style_prop_name="styles",
                include_class_name=False,
            )
            return f"{indent_str}<MetricCardBlock{props} />"

        # Agent component
        if comp_type == "agent-component":
            context.imports.add("AgentComponent")
            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "agent-name": node.get("agent-name") or node.get("agent_name") or "",
                    "agent-title": node.get("agent-title") or node.get("agent_title") or "BESSER Agent",
                },
                style_prop_name="styles",
                include_class_name=False,
            )
            return f"{indent_str}<AgentComponent{props} />"

        # Fallback
        tag = self._select_container_tag(node.get("tag"))
        props = self._build_element_props(component_id, class_list, style, attributes, None)
        if children:
            children_jsx = self._render_component_list(children, selector_style_map, context, indent + 2)
            return (
                f"{indent_str}<{tag}{props}>\n"
                + children_jsx
                + f"\n{indent_str}</{tag}>"
            )
        return f"{indent_str}<{tag}{props} />"

    def _render_button(
        self,
        node: Dict[str, Any],
        component_id: str,
        class_list: List[str],
        style: Dict[str, Any],
        attributes: Dict[str, Any],
        context: _PageRenderContext,
        indent_str: str,
    ) -> str:
        action_type = node.get("action_type")
        if action_type == "run-method" or "endpoint" in attributes:
            context.imports.add("MethodButton")
            endpoint = attributes.get("endpoint")
            is_instance_method = attributes.get("is-instance-method")
            instance_source = attributes.get("instance-source")
            input_params = attributes.get("input-parameters") or {}
            parameters = [
                {"name": name, "type": str(param_type), "required": True}
                for name, param_type in input_params.items()
            ]
            label = attributes.get("button-label") or node.get("label") or node.get("name") or "Execute"

            props = self._build_component_props(
                component_id=component_id,
                class_list=class_list,
                style=style,
                extra_props={
                    "endpoint": endpoint,
                    "label": label,
                    "parameters": parameters if parameters else None,
                    "isInstanceMethod": self._to_bool(is_instance_method),
                    "instanceSourceTableId": instance_source,
                },
            )
            return f"{indent_str}<MethodButton{props} />"

        target_path = self._resolve_target_path(node)
        on_click_expr = None
        if target_path:
            context.needs_navigate = True
            on_click_expr = f"() => navigate({json.dumps(target_path)})"

        props = self._build_element_props(component_id, class_list, style, attributes, on_click_expr)
        label = node.get("label") or node.get("name") or "Button"
        content_expr = f"{{{json.dumps(label, ensure_ascii=False)}}}"
        return f"{indent_str}<button{props}>{content_expr}</button>"

    # ------------------------------------------------------------------ #
    # Prop helpers
    # ------------------------------------------------------------------ #
    def _build_element_props(
        self,
        component_id: str,
        class_list: List[str],
        style: Dict[str, Any],
        attributes: Dict[str, Any],
        on_click_expr: Optional[str],
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> str:
        props: List[str] = []
        if component_id:
            props.append(f" id={json.dumps(component_id)}")

        class_names = [cls[1:] if cls.startswith(".") else cls for cls in (class_list or []) if cls]
        if class_names:
            props.append(f" className={json.dumps(' '.join(class_names))}")

        if style:
            props.append(f" {self._format_prop('style', style)}")

        if on_click_expr:
            props.append(" onClick={" + on_click_expr + "}")

        cleaned_attrs = dict(attributes or {})
        for key in ("id", "class", "className", "style"):
            cleaned_attrs.pop(key, None)

        if extra_props:
            for key, value in extra_props.items():
                if value is not None and value != "":
                    props.append(f" {self._format_prop(key, value)}")

        if cleaned_attrs:
            props.append(f" {{...{json.dumps(cleaned_attrs, ensure_ascii=False)}}}")

        return "".join(props)

    def _build_component_props(
        self,
        component_id: str,
        class_list: List[str],
        style: Dict[str, Any],
        extra_props: Dict[str, Any],
        style_prop_name: str = "style",
        include_class_name: bool = True,
    ) -> str:
        props: List[str] = []
        if component_id:
            props.append(f" id={json.dumps(component_id)}")

        if include_class_name:
            class_names = [cls[1:] if cls.startswith(".") else cls for cls in (class_list or []) if cls]
            if class_names:
                props.append(f" className={json.dumps(' '.join(class_names))}")

        if style:
            props.append(f" {self._format_prop(style_prop_name, style)}")

        for key, value in (extra_props or {}).items():
            if value is None or value == "":
                continue
            props.append(f" {self._format_prop(key, value)}")

        return "".join(props)

    @staticmethod
    def _format_prop(name: str, value: Any) -> str:
        if isinstance(value, bool):
            return f"{name}={{{str(value).lower()}}}"
        if isinstance(value, (int, float)):
            return f"{name}={{{value}}}"
        if isinstance(value, (list, dict)):
            return f"{name}={{{json.dumps(value, ensure_ascii=False)}}}"
        return f"{name}={json.dumps(value, ensure_ascii=False)}"

    # ------------------------------------------------------------------ #
    # Style helpers for page rendering
    # ------------------------------------------------------------------ #
    def _compute_style(
        self,
        component_id: str,
        class_list: List[str],
        selector_style_map: Dict[str, Dict[str, Any]],
        has_children: bool,
    ) -> Dict[str, Any]:
        style: Dict[str, Any] = {}
        if component_id:
            selector = f"#{component_id}"
            if selector in selector_style_map:
                style.update(selector_style_map[selector])

        for cls in class_list or []:
            selector = cls if cls.startswith((".", "#")) else f".{cls}"
            if selector in selector_style_map:
                style.update(selector_style_map[selector])

        if "gjs-cell" in (class_list or []) and has_children:
            if "height" in style:
                style["height"] = "auto"

        if "gjs-row" in (class_list or []):
            if style.get("display") == "table":
                style["display"] = "flex"
                style.setdefault("flexWrap", "wrap")

        if "gjs-cell" in (class_list or []):
            if style.get("display") in ("table-cell", "block"):
                style.pop("display", None)
            width = style.get("width")
            if isinstance(width, str) and width.strip().endswith("%"):
                style.pop("width", None)
            style.setdefault("flex", "1 1 calc(33.333% - 20px)")
            style.setdefault("minWidth", "250px")

        return style

    # ------------------------------------------------------------------ #
    # Navigation helpers
    # ------------------------------------------------------------------ #
    def _resolve_target_path(self, node: Dict[str, Any]) -> Optional[str]:
        if node.get("target_screen_path"):
            return node.get("target_screen_path")
        target_screen_id = node.get("target_screen_id")
        if target_screen_id:
            mapped = getattr(self, "_page_path_map", {}).get(target_screen_id)
            if mapped:
                return mapped

        events = node.get("events") or []
        for event in events:
            actions = event.get("actions") or []
            for action in actions:
                if action.get("kind") == "Transition":
                    target_path = action.get("target_screen_path")
                    if not target_path:
                        target_id = action.get("target_screen_id") or action.get("target_screen")
                        if target_id:
                            target_path = getattr(self, "_page_path_map", {}).get(target_id)
                    return target_path or self._path_from_name(action.get("target_screen"))
        return None

    @staticmethod
    def _path_from_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return "/" + str(name).strip().lower().replace(" ", "-")

    # ------------------------------------------------------------------ #
    # Tag helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_container_tag(tag: Optional[str]) -> str:
        valid_tags = {
            "section",
            "article",
            "header",
            "footer",
            "nav",
            "aside",
            "main",
            "div",
            "ul",
            "ol",
            "li",
            "span",
        }
        tag_value = (tag or "").lower()
        return tag_value if tag_value in valid_tags else "div"

    @staticmethod
    def _select_text_tag(tag: Optional[str]) -> str:
        valid_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "div", "strong", "em", "li"}
        tag_value = (tag or "").lower()
        return tag_value if tag_value in valid_tags else "p"

    # ------------------------------------------------------------------ #
    # Utility helpers for pages
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pascal_case(value: str) -> str:
        if not value:
            return ""
        parts = re.split(r"[^a-zA-Z0-9]+", value)
        return "".join(part.capitalize() for part in parts if part)

    @staticmethod
    def _to_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    # ------------------------------------------------------------------ #
    # App.tsx generation for page routing
    # ------------------------------------------------------------------ #
    def _write_app_tsx(self, page_infos: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        if not page_infos:
            return

        default_route = None
        main_page_id = meta.get("main_page_id")
        for page in page_infos:
            if main_page_id and page.get("page_id") == main_page_id:
                default_route = page.get("route_path")
                break
        if not default_route:
            main_page = next((p for p in page_infos if p.get("is_main")), None)
            default_route = (main_page or page_infos[0]).get("route_path")

        import_lines = []
        for page in page_infos:
            import_lines.append(
                f"import {page['component_name']} from \"{page['file_name']}\";"
            )
        imports = "\n".join(import_lines)

        routes = "\n".join(
            [
                f"            <Route path={json.dumps(p['route_path'])} element={{<{p['component_name']} />}} />"
                for p in page_infos
            ]
        )

        app_contents = (
            "import React from \"react\";\n"
            "import { Routes, Route, Navigate } from \"react-router-dom\";\n"
            "import { TableProvider } from \"./contexts/TableContext\";\n"
            + imports
            + "\n\n"
            "function App() {\n"
            "  return (\n"
            "    <TableProvider>\n"
            "      <div className=\"app-container\">\n"
            "        <main className=\"app-main\">\n"
            "          <Routes>\n"
            + routes
            + "\n"
            f"            <Route path=\"/\" element={{<Navigate to={json.dumps(default_route)} replace />}} />\n"
            f"            <Route path=\"*\" element={{<Navigate to={json.dumps(default_route)} replace />}} />\n"
            "          </Routes>\n"
            "        </main>\n"
            "      </div>\n"
            "    </TableProvider>\n"
            "  );\n"
            "}\n"
            "export default App;\n"
        )

        app_path = self.build_generation_path("src/App.tsx")
        with open(app_path, "w", encoding="utf-8") as f:
            f.write(app_contents)
    # --------------------------------------------------------------------- #
    # Context builders
    # --------------------------------------------------------------------- #

