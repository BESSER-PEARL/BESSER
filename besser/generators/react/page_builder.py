from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set


class _PageRenderContext:
    def __init__(self) -> None:
        self.imports: Set[str] = set()
        self.needs_navigate: bool = False
        # Named input components imported from InputComponents
        self.input_components: Set[str] = set()
        # Alert component
        self.needs_alert_block: bool = False


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

        # Specialized input / alert components
        all_input_components: List[str] = sorted(context.input_components)
        if context.needs_alert_block:
            all_input_components = ["AlertBlock"] + all_input_components
        if all_input_components:
            named = ", ".join(all_input_components)
            imports.append(f"import {{ {named} }} from \"../components/InputComponents\";")

        # useState is needed for complex input components
        if context.input_components or context.needs_alert_block:
            imports.insert(0, "import { useState } from \"react\";")

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
            return self._render_input(node, component_id, class_list, style, attributes, indent_str, context)

        # Alert
        if comp_type == "alert":
            return self._render_alert(node, component_id, class_list, style, attributes, indent_str, context)

        # Form
        if comp_type == "form":
            on_submit_expr = "(e) => { e.preventDefault(); }"
            props = self._build_element_props(component_id, class_list, style, attributes, None)
            props += " onSubmit={" + on_submit_expr + "}"
            inputs = node.get("inputs") or []
            input_lines = []
            for field in inputs:
                input_lines.append(self._render_form_field(field, indent_str + "  "))
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

    def _render_alert(
        self,
        node: Dict[str, Any],
        component_id: str,
        class_list: List[str],
        style: Dict[str, Any],
        attributes: Dict[str, Any],
        indent_str: str,
        context: Optional[_PageRenderContext] = None,
    ) -> str:
        if context is not None:
            context.needs_alert_block = True

        severity = (node.get("severity") or "Info").lower()
        title = node.get("title")
        content = node.get("content") or ""
        dismissible = node.get("dismissible", False)

        props = self._build_component_props(
            component_id=component_id,
            class_list=class_list,
            style=style,
            extra_props={
                "severity": severity,
                "title": title,
                "content": content,
                "dismissible": dismissible or None,
            },
        )
        return f"{indent_str}<AlertBlock{props} />"

    def _render_input(
        self,
        node: Dict[str, Any],
        component_id: str,
        class_list: List[str],
        style: Dict[str, Any],
        attributes: Dict[str, Any],
        indent_str: str,
        context: Optional[_PageRenderContext] = None,
    ) -> str:
        input_type = node.get("input_type") or attributes.get("type") or "Text"
        label = node.get("label") or ""
        placeholder = node.get("placeholder") or attributes.get("placeholder") or ""
        required = node.get("required", False)
        min_value = node.get("min_value")
        max_value = node.get("max_value")
        step = node.get("step")
        multiple = node.get("multiple", False)
        options = node.get("options") or []
        default_value = node.get("default_value")

        base_attrs = dict(attributes)
        base_attrs.pop("type", None)
        base_attrs.pop("placeholder", None)

        def ep(**kw: Any) -> Dict[str, Any]:
            return {k: v for k, v in kw.items() if v is not None and v is not False and v != ""}

        input_style: Dict[str, Any] = {
            "padding": "6px 10px", "border": "1px solid #cbd5e1",
            "borderRadius": "4px", "fontFamily": "inherit",
            "width": "100%", "boxSizing": "border-box",
            **style,
        }

        def with_label(inner_jsx: str) -> str:
            """Wrap input in label+div when a label is present."""
            if not label:
                return inner_jsx
            lbl_style = self._format_prop("style", {"fontSize": "0.875em", "fontWeight": 500, "color": "#374151"})
            wrap_style = self._format_prop("style", {"display": "flex", "flexDirection": "column", "gap": "4px"})
            req_span = ""
            if required:
                req_sty = self._format_prop("style", {"color": "#ef4444", "marginLeft": 2})
                req_span = f'<span {req_sty}>*</span>'
            return (
                f'{indent_str}<div {wrap_style}>\n'
                f'{indent_str}  <label htmlFor={json.dumps(component_id)} {lbl_style}>'
                f'{{{json.dumps(label)}}}{req_span}</label>\n'
                + inner_jsx + "\n"
                + f'{indent_str}</div>'
            )

        # ── Components from InputComponents.tsx ──────────────────────────────

        # Toggle — ToggleInput handles its own label inline
        if input_type == "Toggle":
            if context is not None:
                context.input_components.add("ToggleInput")
            checked = str(default_value).lower() in ("true", "1", "yes") if default_value else False
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(
                    name=component_id or None,
                    label=label or None,
                    defaultChecked=checked or None,
                    required=required or None,
                ),
            )
            return f"{indent_str}<ToggleInput{props} />"

        # Slider — shows live value next to thumb
        if input_type == "Slider":
            if context is not None:
                context.input_components.add("SliderInput")
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(
                    name=component_id or None,
                    min=min_value, max=max_value, step=step,
                    defaultValue=default_value,
                ),
            )
            return with_label(f"{indent_str}  <SliderInput{props} />") if label else f"{indent_str}<SliderInput{props} />"

        # Rating — interactive star buttons
        if input_type == "Rating":
            if context is not None:
                context.input_components.add("RatingInput")
            max_stars = int(max_value) if max_value is not None else 5
            dv = int(default_value) if default_value is not None else None
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(name=component_id or None, maxStars=max_stars, defaultValue=dv),
            )
            return with_label(f"{indent_str}  <RatingInput{props} />") if label else f"{indent_str}<RatingInput{props} />"

        # Tags — chip-based text input
        if input_type == "Tags":
            if context is not None:
                context.input_components.add("TagsInput")
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(
                    name=component_id or None,
                    placeholder=placeholder or None,
                    required=required or None,
                ),
            )
            return with_label(f"{indent_str}  <TagsInput{props} />") if label else f"{indent_str}<TagsInput{props} />"

        # OTP — segmented digit boxes
        if input_type == "OTP":
            if context is not None:
                context.input_components.add("OTPInput")
            length = int(max_value) if max_value is not None else 6
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(name=component_id or None, length=length),
            )
            return with_label(f"{indent_str}  <OTPInput{props} />") if label else f"{indent_str}<OTPInput{props} />"

        # DateRange — two linked date pickers
        if input_type == "DateRange":
            if context is not None:
                context.input_components.add("DateRangeInput")
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(
                    name=component_id or None,
                    required=required or None,
                ),
            )
            return with_label(f"{indent_str}  <DateRangeInput{props} />") if label else f"{indent_str}<DateRangeInput{props} />"

        # MultiSelect — click-to-toggle chip list
        if input_type == "MultiSelect":
            if context is not None:
                context.input_components.add("MultiSelectInput")
            dv_list: List[str] = []
            if isinstance(default_value, str) and default_value:
                dv_list = [s.strip() for s in default_value.split(",") if s.strip()]
            props = self._build_component_props(
                component_id=component_id, class_list=class_list, style=style,
                extra_props=ep(
                    name=component_id or None,
                    options=options or None,
                    defaultValue=dv_list or None,
                    required=required or None,
                ),
            )
            return with_label(f"{indent_str}  <MultiSelectInput{props} />") if label else f"{indent_str}<MultiSelectInput{props} />"

        # ── Pure-HTML types ───────────────────────────────────────────────────

        # TextArea / RichText
        if input_type in {"TextArea", "RichText"}:
            textarea_style = {**input_style, "minHeight": "80px", "resize": "vertical"}
            props = self._build_element_props(
                component_id, class_list, textarea_style, base_attrs, None,
                extra_props=ep(placeholder=placeholder or None, required=required or None),
            )
            inner = f"{indent_str}<textarea{props}></textarea>"
            return with_label(inner) if label else inner

        # Dropdown
        if input_type == "Dropdown":
            props = self._build_element_props(
                component_id, class_list, input_style, base_attrs, None,
                extra_props=ep(required=required or None),
            )
            placeholder_opt = f'{indent_str}  <option value="">— Select —</option>\n' if not required else ""
            opt_lines = "\n".join(
                f'{indent_str}  <option value={json.dumps(o.get("value", ""))}>'
                f'{{{json.dumps(o.get("label", ""))}}}</option>'
                for o in options
            )
            inner = (
                f"{indent_str}<select{props}>\n"
                + placeholder_opt
                + opt_lines
                + f"\n{indent_str}</select>"
            )
            return with_label(inner) if label else inner

        # RadioGroup
        if input_type == "RadioGroup":
            row_style = self._format_prop("style", {"display": "flex", "alignItems": "center", "gap": "6px"})
            grp_style = self._format_prop("style", {"display": "flex", "flexDirection": "column", "gap": "6px"})
            lines = [f'{indent_str}<div role="radiogroup" id={json.dumps(component_id)} {grp_style}>']
            for o in options:
                lines.append(
                    f'{indent_str}  <label {row_style}>'
                    f'<input type="radio" name={json.dumps(component_id)} value={json.dumps(o.get("value", ""))} />'
                    f' {{{json.dumps(o.get("label", ""))}}}</label>'
                )
            lines.append(f"{indent_str}</div>")
            inner = "\n".join(lines)
            return with_label(inner) if label else inner

        # CheckboxGroup
        if input_type == "CheckboxGroup":
            row_style = self._format_prop("style", {"display": "flex", "alignItems": "center", "gap": "6px"})
            grp_style = self._format_prop("style", {"display": "flex", "flexDirection": "column", "gap": "6px"})
            lines = [f'{indent_str}<div id={json.dumps(component_id)} {grp_style}>']
            for o in options:
                lines.append(
                    f'{indent_str}  <label {row_style}>'
                    f'<input type="checkbox" name={json.dumps(component_id)} value={json.dumps(o.get("value", ""))} />'
                    f' {{{json.dumps(o.get("label", ""))}}}</label>'
                )
            lines.append(f"{indent_str}</div>")
            inner = "\n".join(lines)
            return with_label(inner) if label else inner

        # Checkbox — inline [box] label layout
        if input_type == "Checkbox":
            checked = str(default_value).lower() in ("true", "1", "yes") if default_value else False
            lbl_style = self._format_prop("style", {"display": "inline-flex", "alignItems": "center", "gap": "6px", "cursor": "pointer", "userSelect": "none", "fontSize": "0.875em", "fontWeight": 500, "color": "#374151"})
            req_span = ""
            if required:
                req_sty = self._format_prop("style", {"color": "#ef4444", "marginLeft": 2})
                req_span = f'<span {req_sty}>*</span>'
            lbl_text = f"{{{json.dumps(label)}}}{req_span}" if label else ""
            inner_props = self._build_element_props(
                component_id, class_list, {}, base_attrs, None,
                extra_props=ep(name=component_id or None, required=required or None, defaultChecked=checked or None),
            )
            return (
                f'{indent_str}<div>\n'
                f'{indent_str}  <label htmlFor={json.dumps(component_id)} {lbl_style}>\n'
                f'{indent_str}    <input type="checkbox"{inner_props} />\n'
                f'{indent_str}    {lbl_text}\n'
                f'{indent_str}  </label>\n'
                f'{indent_str}</div>'
            )

        # Range
        if input_type == "Range":
            props = self._build_element_props(
                component_id, class_list, {"flex": "1", **style}, base_attrs, None,
                extra_props=ep(min=min_value, max=max_value, step=step, defaultValue=default_value),
            )
            inner = f'{indent_str}<input type="range"{props} />'
            return with_label(inner) if label else inner

        # Spinner (number with step)
        if input_type == "Spinner":
            props = self._build_element_props(
                component_id, class_list, input_style, base_attrs, None,
                extra_props=ep(
                    min=min_value, max=max_value,
                    step=step if step is not None else 1,
                    required=required or None,
                    defaultValue=default_value,
                ),
            )
            inner = f'{indent_str}<input type="number"{props} />'
            return with_label(inner) if label else inner

        # ImageUpload
        if input_type == "ImageUpload":
            props = self._build_element_props(
                component_id, class_list, input_style, base_attrs, None,
                extra_props=ep(accept="image/*", multiple=multiple or None, required=required or None),
            )
            inner = f'{indent_str}<input type="file"{props} />'
            return with_label(inner) if label else inner

        # File
        if input_type == "File":
            props = self._build_element_props(
                component_id, class_list, input_style, base_attrs, None,
                extra_props=ep(multiple=multiple or None, required=required or None),
            )
            inner = f'{indent_str}<input type="file"{props} />'
            return with_label(inner) if label else inner

        # Color — small square swatch
        if input_type == "Color":
            color_style = {"width": "48px", "height": "36px", "padding": "2px",
                           "border": "1px solid #cbd5e1", "borderRadius": "4px", "cursor": "pointer"}
            props = self._build_element_props(
                component_id, class_list, color_style, base_attrs, None,
                extra_props=ep(defaultValue=default_value),
            )
            inner = f'{indent_str}<input type="color"{props} />'
            return with_label(inner) if label else inner

        # DateTime
        if input_type == "DateTime":
            props = self._build_element_props(
                component_id, class_list, input_style, base_attrs, None,
                extra_props=ep(required=required or None),
            )
            inner = f'{indent_str}<input type="datetime-local"{props} />'
            return with_label(inner) if label else inner

        # Remaining types: direct HTML input type mapping
        html_type_map = {
            "Text": "text", "Email": "email", "Number": "number", "Password": "password",
            "Date": "date", "Time": "time", "URL": "url", "Tel": "tel",
            "Search": "search", "Hidden": "hidden",
        }
        html_type = html_type_map.get(input_type, str(input_type).lower())
        extra_kw = ep(placeholder=placeholder or None, required=required or None, defaultValue=default_value)
        if input_type == "Number":
            extra_kw.update(ep(min=min_value, max=max_value, step=step))
        props = self._build_element_props(
            component_id, class_list,
            {} if input_type == "Hidden" else input_style,
            base_attrs, None,
            extra_props=extra_kw or None,
        )
        inner = f'{indent_str}<input type={json.dumps(html_type)}{props} />'
        return with_label(inner) if label else inner

    def _render_form_field(self, field: Dict[str, Any], indent_str: str) -> str:
        field_id = field.get("id") or ""
        label = field.get("label") or field_id
        field_type = field.get("type") or "Text"
        placeholder = field.get("placeholder") or ""
        required = field.get("required", False)
        options = field.get("options") or []
        min_value = field.get("min_value")
        max_value = field.get("max_value")
        step = field.get("step")
        multiple = field.get("multiple", False)
        default_value = field.get("default_value")

        wrapper_style = self._format_prop("style", {"marginBottom": "16px"})
        label_style = self._format_prop("style", {"display": "block", "marginBottom": "5px", "fontWeight": "500"})

        label_inner = f"{{{json.dumps(label, ensure_ascii=False)}}}"
        if required:
            req_style = self._format_prop("style", {"color": "#ef4444", "marginLeft": "2px"})
            label_inner += f"<span {req_style}>*</span>"

        label_jsx = f"{indent_str}<label htmlFor={json.dumps(field_id)} {label_style}>{label_inner}</label>"

        field_node: Dict[str, Any] = {
            "input_type": field_type,
            "placeholder": placeholder,
            "required": required,
            "options": options,
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "multiple": multiple,
            "default_value": default_value,
        }
        input_jsx = self._render_input(field_node, field_id, [], {}, {}, indent_str)

        return (
            f"{indent_str}<div {wrapper_style}>\n"
            + label_jsx + "\n"
            + input_jsx + "\n"
            + f"{indent_str}</div>"
        )

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
            parameters: List[Dict[str, Any]] = []
            if isinstance(input_params, dict):
                for name, param_meta in input_params.items():
                    if isinstance(param_meta, dict):
                        parameter: Dict[str, Any] = {
                            "name": name,
                            "type": str(param_meta.get("type", "any")),
                            "required": bool(param_meta.get("required", True)),
                        }
                        if "default" in param_meta:
                            parameter["default"] = param_meta.get("default")
                        if param_meta.get("input_kind"):
                            parameter["inputKind"] = str(param_meta.get("input_kind"))
                        if param_meta.get("entity"):
                            parameter["entity"] = str(param_meta.get("entity"))
                        if param_meta.get("lookup_field"):
                            parameter["lookupField"] = str(param_meta.get("lookup_field"))
                        if isinstance(param_meta.get("options"), list):
                            parameter["options"] = param_meta.get("options")
                        parameters.append(parameter)
                    else:
                        parameters.append({"name": name, "type": str(param_meta), "required": True})
            elif isinstance(input_params, list):
                for item in input_params:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not name:
                        continue
                    parameter: Dict[str, Any] = {
                        "name": name,
                        "type": str(item.get("type", "any")),
                        "required": bool(item.get("required", True)),
                    }
                    if "default" in item:
                        parameter["default"] = item.get("default")
                    if item.get("inputKind"):
                        parameter["inputKind"] = str(item.get("inputKind"))
                    if item.get("entity"):
                        parameter["entity"] = str(item.get("entity"))
                    if item.get("lookupField"):
                        parameter["lookupField"] = str(item.get("lookupField"))
                    if isinstance(item.get("options"), list):
                        parameter["options"] = item.get("options")
                    parameters.append(parameter)
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

