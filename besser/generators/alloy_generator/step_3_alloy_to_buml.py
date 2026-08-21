"""
Script completo para integrar diagrama de clases BUML con diagrama de objetos generado desde Alloy XML.

Este script:
1. Lee el archivo XML de instancia generado por Alloy Analyzer
2. Lee el modelo BUML original (diagrama de clases)
3. Genera el diagrama de objetos usando el conversor AlloyToBesserConverter
4. Combina ambos en un archivo BUML completo con clase + objetos + project
"""

import logging
import os
import re
import sys
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__) 



class AlloyToBesserConverter:
    """Convierte instancias Alloy XML a objetos BESSER."""
    
    def __init__(self, xml_file: str):
        """
        Inicializa el conversor.
        
        Args:
            xml_file: Ruta al archivo XML de Alloy
        """
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        
        # Estructuras de datos para almacenar la información parseada
        self.signatures = {}  # sig_id -> {label, atoms}
        self.fields = {}  # field_id -> {label, parent_id, tuples}
        self.atoms_by_sig = {}  # sig_label -> [atoms]
        self.builtin_sigs = {'seq/Int', 'Int', 'String', 'univ', 'boolean/Bool', 
                            'boolean/True', 'boolean/False'}
        
    def parse_xml(self):
        """Parsea el archivo XML y extrae signatures, fields y atoms."""
        
        # Parsear signatures
        for sig in self.root.findall('.//sig'):
            sig_id = sig.get('ID')
            sig_label = sig.get('label')
            parent_id = sig.get('parentID')
            
            # Ignorar signatures built-in
            if sig_label in self.builtin_sigs:
                continue
                
            atoms = [atom.get('label') for atom in sig.findall('atom')]
            
            self.signatures[sig_id] = {
                'label': sig_label,
                'atoms': atoms,
                'builtin': sig.get('builtin') == 'yes',
                'parent_id': parent_id,
            }
            
            # Organizar atoms por signature
            if not self.signatures[sig_id]['builtin']:
                self.atoms_by_sig[sig_label] = atoms
                #print(atoms)
        
        # Parsear fields
        for field in self.root.findall('.//field'):
            field_id = field.get('ID')
            field_label = field.get('label')
            parent_id = field.get('parentID')
            
            tuples = []
            for tuple_elem in field.findall('tuple'):
                atoms = [atom.get('label') for atom in tuple_elem.findall('atom')]
                if len(atoms) >= 2:
                    tuples.append((atoms[0], atoms[1]))
            
            self.fields[field_id] = {
                'label': field_label,
                'parent_id': parent_id,
                'tuples': tuples
            }
            #print(self.fields[field_id])


        # =========================================================
        # --- MOSTRAR EL ÁRBOL EXTRAÍDO EN LOS LOGS DEL SERVIDOR ---
        # =========================================================
        
        # Convertimos a JSON formateado con sangría de 2 espacios
        #arbol_str = json.dumps(arbol_completo, indent=2, ensure_ascii=False)
        
        #logger.warning("=== ESTRUCTURA EXTRAÍDA DEL XML ===\n%s\n===================================", arbol_str)



    
    def get_class_name(self, sig_label: str) -> str:
        """
        Extrae el nombre de la clase de una signature label.
        
        Args:
            sig_label: Label de la signature (ej: 'this/Player')
        
        Returns:
            Nombre de la clase (ej: 'Player')
        """
        if '/' in sig_label:
            return sig_label.split('/')[-1]
        return sig_label
    
    def remove_class_prefix(self, field_name: str, class_name: str) -> str:
        """
        Remueve el prefijo de clase de un nombre de campo.
        
        Args:
            field_name: Nombre del campo (ej: 'Player_name')
            class_name: Nombre de la clase (ej: 'Player')
        
        Returns:
            Nombre del campo sin prefijo (ej: 'name')
        """
        prefix = f"{class_name}_"
        if field_name.startswith(prefix):
            return field_name[len(prefix):]
        elif field_name.__contains__("_"):
            return field_name[field_name.index("_")+1:]
        return field_name

    def is_enum_value(self, atom_label: str) -> bool:
        """Determina si un atom es un valor de enumeración."""
        return atom_label.startswith('ENUM_')
    
    def get_enum_value(self, atom_label: str) -> str:
        """
        Extrae el valor de enumeración de un atom.
        
        Args:
            atom_label: Label del atom (ej: 'ENUM_Position_CENTER$0')
        
        Returns:
            Valor de enumeración (ej: 'CENTER')
        """
        # Formato: ENUM_EnumName_VALUE$n
        parts = atom_label.split('_')
        if len(parts) >= 3:
            value = '_'.join(parts[2:])  # Tomar todo después de EnumName
            # Remover el sufijo $n
            if '$' in value:
                value = value.split('$')[0]
            return value
        return atom_label
    
    def is_primitive_type(self, atom_label: str) -> bool:
        """Determina si un atom representa un tipo primitivo."""
        try:
            int(atom_label)
            return True
        except ValueError:
            pass
        return '$' in atom_label
    
    def get_primitive_value(self, atom_label: str, atom_type: str | None = None) -> any:
        """
        Extrae el valor primitivo de un atom.
        
        Args:
            atom_label: Label del atom
            atom_type: Tipo esperado (Int, String, etc.)
        
        Returns:
            Valor primitivo convertido
        """
        # Enteros
        try:
            return int(atom_label)
        except ValueError:
            pass
        
        # Strings - retornar el identificador sin el sufijo
        if '$' in atom_label:
            base_name = atom_label.split('$')[0]
            return f'"{base_name}"'  # Return as quoted string
        
        return f'"{atom_label}"'
    
    DATE_SIG_PATTERN = re.compile(r"^d\d{8}$")
    
    def _date_sig_label(self) -> str | None:
        """Devuelve la label de la sig 'date' (ej: 'this/date') si existe."""
        for sig_label in self.atoms_by_sig:
            if self.get_class_name(sig_label) == "date":
                return sig_label
        return None
    
    def is_date_value(self, atom_label: str) -> bool:
        """Determina si un atom es un valor de fecha.

        Reconoce tanto los literales ``dMMDDYYYY`` (one sigs emitidos por las
        constraints OCL) como los atoms libres de la sig ``date`` generados por
        Alloy (ej: ``date$0``, ``date$01`` u otros strings).
        """
        base = atom_label.split("$")[0]
        if self.DATE_SIG_PATTERN.match(base):
            return True
        date_sig_label = self._date_sig_label()
        return bool(date_sig_label and atom_label in self.atoms_by_sig[date_sig_label])
    
    def get_date_value(self, atom_label: str) -> str:
        """
        Extrae el valor de fecha de un atom.

        Los literales ``dMMDDYYYY`` se decodifican a 'DD-MM-YYYY'; el resto de
        atoms de la sig ``date`` se muestran tal cual los encontró Alloy, entre
        comillas (ej: '"date$0"').
        
        Args:
            atom_label: Label del atom (ej: 'd01012000$0' o 'date$01')
        
        Returns:
            Valor de fecha como string con comillas (ej: '"01-01-2000"')
        """
        base = atom_label.split("$")[0]
        if self.DATE_SIG_PATTERN.match(base):
            return f'"{base[3:5]}-{base[1:3]}-{base[5:9]}"'
        return f'"{atom_label}"'
    
    def is_domain_class_name(self, class_name: str) -> bool:
        """Determina si *class_name* es una clase de dominio del usuario."""
        if not class_name:
            return False
        if self.is_enum_value(class_name) or class_name.startswith("ENUM_"):
            return False
        if class_name in ("str", "Bool", "True", "False", "date", "Ord"):
            return False
        return not self.DATE_SIG_PATTERN.match(class_name)
    
    def is_object_reference(self, atom_label: str) -> bool:
        """
        Determina si un atom es una referencia a otro objeto del dominio.
        
        Args:
            atom_label: Label del atom a verificar
        
        Returns:
            True si es una referencia a objeto, False si es primitivo
        """
        # Los objetos del dominio tienen el formato ClassName$N
        # Ejemplos: Player$0, Team$1, City$0, Fan$3
        if '$' in atom_label:
            base = atom_label.split('$')[0]
            
            # Excluir tipos especiales que no son objetos del dominio
            if base in ['str', 'pepe', 'Position'] or base.startswith('ENUM_'):
                return False
            
            # Verificar si existe en atoms_by_sig con el prefijo this/
            for sig_label in self.atoms_by_sig:
                class_name = self.get_class_name(sig_label)
                if (class_name == base
                        and atom_label in self.atoms_by_sig[sig_label]
                        and class_name not in ['str', 'Bool', 'True', 'False']):
                    return True
        return False
    


    def get_fields_for_signature(self, sig_label: str) -> dict[str, list[tuple]]:
            sig_id = None
            for sid, sig_data in self.signatures.items():
                if sig_data['label'] == sig_label:
                    sig_id = sid
                    break

            if not sig_id:
                return {}

            fields_dict = {}
            for field_data in self.fields.values():
                if field_data['parent_id'] == sig_id:          # ← SOLO esto, sin else
                    field_name = field_data['label']
                    fields_dict[field_name] = field_data['tuples']

            return fields_dict


    def _pair_association_fields(self) -> dict[str, frozenset]:
        """Empareja los dos extremos de una misma asociación bidireccional.

        Alloy materializa una asociación bidireccional como dos campos con
        conjuntos de tuplas exactamente transpuestos. Este mapa permite
        deduplicar solo esas dos mitades sin colapsar asociaciones distintas
        entre el mismo par de objetos.
        """
        field_tuples: dict[str, set[tuple]] = {}
        for field_data in self.fields.values():
            label = field_data['label']
            field_tuples[label] = set(field_data['tuples'])

        used: set[str] = set()
        paired: dict[str, frozenset] = {}
        for field_label, tuples in field_tuples.items():
            if field_label in used or not tuples:
                continue

            transposed = {(to_atom, from_atom) for from_atom, to_atom in tuples}
            partner = None
            for other_label, other_tuples in field_tuples.items():
                if other_label in used or other_label == field_label or not other_tuples:
                    continue
                if other_tuples == transposed:
                    partner = other_label
                    break

            if partner:
                assoc = frozenset([field_label, partner])
                paired[field_label] = assoc
                paired[partner] = assoc
                used.update([field_label, partner])
            else:
                paired[field_label] = frozenset([field_label])

        return paired






    def generate_object_diagram_code(self) -> str:
        """Genera código BUML para el diagrama de objetos derivado del XML.

        Solo instancia la clase concreta más específica de cada átomo,
        incorpora atributos heredados y conserva asociaciones múltiples entre
        las mismas clases u objetos.
        """
        code_lines = []

        # Identificar las clases del dominio. Las signatures de enumeración no
        # deben materializarse como objetos.
        domain_classes = set()
        for sig_label, atoms in self.atoms_by_sig.items():
            if not sig_label.startswith("this/"):
                continue
            class_name = self.get_class_name(sig_label)
            if not self.is_domain_class_name(class_name):
                continue
            if atoms and all(self.is_enum_value(atom) for atom in atoms):
                continue
            domain_classes.add(class_name)

        def signature_depth(sig_id: str | None) -> int:
            depth = 0
            current_id = sig_id
            while current_id and current_id in self.signatures:
                parent_id = self.signatures[current_id].get('parent_id')
                if not parent_id or parent_id not in self.signatures:
                    break
                depth += 1
                current_id = parent_id
            return depth

        domain_signatures = []
        for sig_id, sig_data in self.signatures.items():
            class_name = self.get_class_name(sig_data['label'])
            if class_name not in domain_classes:
                continue
            domain_signatures.append({
                'class_name': class_name,
                'atoms': set(sig_data['atoms']),
                'depth': signature_depth(sig_id),
            })

        def leaf_class_for(atom_label: str) -> str | None:
            """Devuelve la clase concreta más específica que contiene el átomo."""
            containing = [sig for sig in domain_signatures if atom_label in sig['atoms']]
            if not containing:
                return None
            leaf_sig = max(
                containing,
                key=lambda sig: (sig['depth'], -len(sig['atoms']), sig['class_name'])
            )
            return leaf_sig['class_name']

        created_objects = {}  # atom_label -> variable_name
        relations = []  # [(from_var, relation_name, to_atom, field_name), ...]

        for sig_label, atoms in self.atoms_by_sig.items():
            class_name = self.get_class_name(sig_label)

            if class_name not in domain_classes:
                continue

            for i, atom_label in enumerate(atoms):
                if leaf_class_for(atom_label) != class_name:
                    continue

                obj_var = f"{class_name.lower()}_{i}_obj"
                obj_name = atom_label.replace('$', '_')

                created_objects[atom_label] = obj_var
                attributes = {}

                # Recorrer todos los fields para incluir también atributos y
                # asociaciones heredadas desde clases ancestro.
                for field_data in self.fields.values():
                    field_name = field_data['label']
                    tuples = field_data['tuples']
                    attr_name = self.remove_class_prefix(field_name, class_name)

                    for tuple_from, tuple_to in tuples:
                        if tuple_from != atom_label:
                            continue

                        if self.is_date_value(tuple_to):
                            attributes[attr_name] = self.get_date_value(tuple_to)
                        elif self.is_object_reference(tuple_to):
                            relations.append((obj_var, attr_name, tuple_to, field_name))
                        elif self.is_enum_value(tuple_to):
                            enum_value = self.get_enum_value(tuple_to)
                            attributes[attr_name] = f'"{enum_value}"'
                        else:
                            attributes[attr_name] = self.get_primitive_value(tuple_to)

                attribute_mapping_parts = []
                for attr_name, attr_value in attributes.items():
                    attribute_mapping_parts.append(f"{attr_name!r}: {attr_value}")

                if attribute_mapping_parts:
                    code_lines.append(
                        f'{obj_var} = {class_name}("{obj_name}").attributes(**{{{", ".join(attribute_mapping_parts)}}}).build()'
                    )
                else:
                    code_lines.append(
                        f'{obj_var} = {class_name}("{obj_name}").build()'
                    )
        
        # Agregar línea en blanco
        code_lines.append("")
        
        # Agregar comentario para relaciones
        if relations:
            code_lines.append("# Establecer relaciones entre objetos")

        paired_fields = self._pair_association_fields()
        seen_links = set()
        deduplicated_relations = []
        for from_var, relation_name, to_atom, field_name in relations:
            if to_atom not in created_objects:
                continue
            to_var = created_objects[to_atom]
            assoc_id = paired_fields.get(field_name, frozenset([field_name]))
            canonical = (frozenset([from_var, to_var]), assoc_id)
            if canonical in seen_links:
                continue
            seen_links.add(canonical)
            deduplicated_relations.append((from_var, relation_name, to_atom))

        relations = deduplicated_relations

        # Agrupar relaciones por (from_var, relation_name) para manejar multiplicidad muchos
        grouped_relations = {}
        for from_var, relation_name, to_atom in relations:
            if to_atom in created_objects:
                key = (from_var, relation_name)
                if key not in grouped_relations:
                    grouped_relations[key] = []
                grouped_relations[key].append(created_objects[to_atom])
        
        # Generar código de relaciones
        for (from_var, relation_name), to_vars in grouped_relations.items():
            unique_targets = sorted(set(to_vars))
            if len(unique_targets) == 1:
                code_lines.append(
                    f"setattr({from_var}, {relation_name!r}, {unique_targets[0]})"
                )
            else:
                targets_expr = ", ".join(unique_targets)
                code_lines.append(
                    f"setattr({from_var}, {relation_name!r}, {{{targets_expr}}})"
                )
        
        # Agregar línea en blanco
        if relations:
            code_lines.append("")
        
        # Crear el ObjectModel
        code_lines.append("# Object Model instance")
        all_objects = ", ".join(created_objects.values())
        code_lines.append("object_model: ObjectModel = ObjectModel(")
        code_lines.append('    name="Object_Diagram",')
        code_lines.append(f"    objects={{{all_objects}}}")
        code_lines.append(")")
        #print()
        #print(code_lines)
        return "\n".join(code_lines)


class BUMLModelIntegrator:
    """Integra un modelo BUML original con un diagrama de objetos generado desde Alloy."""
    
    def __init__(self, original_buml_file: str, xml_instance_file: str):
        """
        Inicializa el integrador.
        
        Args:
            original_buml_file: Ruta al archivo BUML original (diagrama de clases)
            xml_instance_file: Ruta al archivo XML de instancia de Alloy
        """
        self.original_buml_file = original_buml_file
        self.xml_instance_file = xml_instance_file
        
        # Leer el contenido del archivo BUML original
        with open(original_buml_file, 'r', encoding='utf-8') as f:
            self.original_content = f.read()
    
    def extract_structural_model_section(self) -> str:
        """
        Extrae la sección del modelo estructural (diagrama de clases).
        
        Returns:
            Código del modelo estructural
        """
        # Buscar desde el inicio hasta antes de cualquier sección de objetos o proyecto
        patterns = [
            r'################\s*\n#\s*OBJECT MODEL\s*#',
            r'##############\s*\n\s*from besser\.BUML\.metamodel\.object',
            r'######################\s*\n#\s*PROJECT DEFINITION\s*#'
        ]
        
        end_pos = len(self.original_content)
        for pattern in patterns:
            match = re.search(pattern, self.original_content, re.IGNORECASE)
            if match:
                end_pos = min(end_pos, match.start())
        
        structural_section = self.original_content[:end_pos].rstrip()
        return structural_section
    
    def extract_project_section(self) -> str:
        """
        Extrae la sección de definición del proyecto si existe.
        
        Returns:
            Código de la sección del proyecto o string vacío
        """
        pattern = r'######################\s*\n#\s*PROJECT DEFINITION\s*#\s*\n######################\s*\n(.*)'
        match = re.search(pattern, self.original_content, re.DOTALL)
        
        if match:
            project_section = match.group(0).strip()
            
            # Modificar la línea de models para incluir object_model
            # Buscar: models=[domain_model] o models=[domain_model, ...]
            # Reemplazar con: models=[domain_model, object_model]
            
            # Patrón para encontrar la línea de models
            models_pattern = r'(models=\[)([^\]]+)(\])'
            
            def replace_models(match):
                prefix = match.group(1)  # "models=["
                models_list = match.group(2).strip()  # contenido actual
                suffix = match.group(3)  # "]"
                
                # Si ya incluye object_model, no modificar
                if 'object_model' in models_list:
                    return match.group(0)
                
                # Agregar object_model a la lista
                if models_list:
                    # Ya hay modelos, agregar object_model
                    return f"{prefix}{models_list}, object_model{suffix}"
                else:
                    # Lista vacía, solo agregar object_model
                    return f"{prefix}object_model{suffix}"
            
            project_section = re.sub(models_pattern, replace_models, project_section)
            
            return project_section
        return ""
    
    def generate_integrated_model(self, output_file: str | None = None) -> str:
        """
        Genera el modelo BUML integrado completo.
        
        Args:
            output_file: Archivo donde guardar el modelo (opcional)
        
        Returns:
            Código del modelo integrado
        """
        # 1. Extraer modelo estructural (diagrama de clases)
        structural_section = self.extract_structural_model_section()
        
        # 2. Generar diagrama de objetos desde XML de Alloy
        converter = AlloyToBesserConverter(self.xml_instance_file)
        converter.parse_xml()
        object_diagram_code = converter.generate_object_diagram_code()
        
        # 3. Extraer sección de proyecto
        project_section = self.extract_project_section()
        
        # 4. Ensamblar el modelo completo
        integrated_lines = []
        
        # Sección estructural
        integrated_lines.append(structural_section)
        integrated_lines.append("")
        integrated_lines.append("")
        
        # Sección de objetos
        integrated_lines.append("################")
        integrated_lines.append("# OBJECT MODEL #")
        integrated_lines.append("################")
        integrated_lines.append("")
        integrated_lines.append("from besser.BUML.metamodel.object import ObjectModel")
        integrated_lines.append("import datetime")
        integrated_lines.append("")
        integrated_lines.append(object_diagram_code)
        integrated_lines.append("")
        integrated_lines.append("")
        
        # Sección de proyecto (si existe)
        if project_section:
            integrated_lines.append(project_section)
        
        integrated_model = "\n".join(integrated_lines)
        
        # Guardar si se especifica archivo de salida
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(integrated_model)
        
        return integrated_model


def main():
    """Función principal para uso como script."""
    
    if len(sys.argv) < 3:
        print("Uso: python alloy_to_buml_complete.py <archivo_buml_original> <archivo_xml_alloy> [archivo_salida]")
        print()
        print("Argumentos:")
        print("  archivo_buml_original : Archivo Python con el modelo BUML original (diagrama de clases)")
        print("  archivo_xml_alloy     : Archivo XML de instancia generado por Alloy Analyzer")
        print("  archivo_salida        : (Opcional) Archivo donde guardar el modelo integrado")
        print()
        print("Ejemplo:")
        print("  python alloy_to_buml_complete.py team3.py instancia.xml team_completo.py")
        sys.exit(1)
    
    original_buml_file = sys.argv[1]
    xml_instance_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validar que los archivos existen
    if not os.path.exists(original_buml_file):
        print(f"Error: El archivo '{original_buml_file}' no existe.")
        sys.exit(1)
    
    if not os.path.exists(xml_instance_file):
        print(f"Error: El archivo '{xml_instance_file}' no existe.")
        sys.exit(1)
    
    # Crear integrador y generar modelo
    #print(f"Integrando modelo BUML...")
    #print(f"  - Modelo original: {original_buml_file}")
    #print(f"  - Instancia XML:   {xml_instance_file}")
    
    integrator = BUMLModelIntegrator(original_buml_file, xml_instance_file)
    integrator.generate_integrated_model(output_file)
    
    #print()
    #print("=" * 80)
    #print("MODELO BUML INTEGRADO GENERADO")
    #print("=" * 80)
    #print(integrated_model)
    #print("=" * 80)
    
    if output_file:
        print(f"\n✓ BUML model + generated object diagram: {output_file}")
   

if __name__ == "__main__":
    main()
