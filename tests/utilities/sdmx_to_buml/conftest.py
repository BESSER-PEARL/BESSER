"""Fixtures for the SDMX-to-B-UML tests: a synthetic structure message and recorded real ones."""

import gzip
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

NS = ('xmlns:m="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message" '
      'xmlns:s="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure" '
      'xmlns:c="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common"')


def _codelist(cid: str, codes: list, parents: dict = None) -> str:
    parents = parents or {}
    items = "".join(
        f'<s:Code id="{c}"><c:Name xml:lang="en">{n}</c:Name>'
        + (f'<s:Parent><Ref id="{parents[c]}"/></s:Parent>' if c in parents else "")
        + "</s:Code>"
        for c, n in codes)
    return (f'<s:Codelist id="{cid}" agencyID="TEST" version="1.0">'
            f'<c:Name xml:lang="en">{cid}</c:Name>{items}</s:Codelist>')


def _dim(did: str, pos: int, cl: str = None) -> str:
    rep = (f'<s:LocalRepresentation><s:Enumeration><Ref id="{cl}" agencyID="TEST" version="1.0" class="Codelist"/>'
           f'</s:Enumeration></s:LocalRepresentation>') if cl else ""
    return (f'<s:Dimension id="{did}" position="{pos}"><s:ConceptIdentity><Ref id="{did}"/></s:ConceptIdentity>'
            f'{rep}</s:Dimension>')


def nomenclature_codes() -> list:
    """A numeric nomenclature nesting by prefix (2, 4, 6 and 8 digits), with a total."""
    codes = [("TOTAL", "Total")]
    for ch in ("01", "84", "85"):
        codes.append((ch, f"Chapter {ch}"))
        for h in range(1, 5):
            heading = f"{ch}{h:02d}"
            codes.append((heading, f"Heading {heading}"))
            for s in range(1, 3):
                sub = f"{heading}{s:02d}"
                codes.append((sub, f"Subheading {sub}"))
                codes.append((f"{sub}00", f"Line {sub}00"))
    return codes


AREAS = ([("WORLD", "All countries of the world"), ("EU27_2020", "European Union - 27 countries"),
          ("BE", "Belgium"), ("DE", "Germany"), ("FR", "France"), ("LU", "Luxembourg"), ("QV", "Not specified")]
         + [(f"X{chr(65 + i)}", f"Country {i}") for i in range(55)])

SECTORS = [("_T", "Total economy"), ("S1", "Total economy (S1)"), ("S11", "Non-financial corporations"),
           ("S12", "Financial corporations"), ("S13", "General government")]


def build_structure_xml(doctype: bool = False) -> bytes:
    codelists = "".join([
        _codelist("CL_FREQ", [("A", "Annual"), ("M", "Monthly")]),
        _codelist("CL_AREA", AREAS),
        _codelist("CL_PRODUCT", nomenclature_codes()),
        _codelist("CL_FLOW", [("1", "IMPORT"), ("2", "EXPORT")]),
        _codelist("CL_RANK", [(str(i), f"Rank {i}") for i in range(1, 61)]),
        _codelist("CL_SECTOR", SECTORS, parents={"S11": "S1", "S12": "S1", "S13": "S1"}),
        _codelist("CL_UNIT", [("EUR", "Euro")]),
    ])
    dims = "".join([
        _dim("FREQ", 1, "CL_FREQ"), _dim("REPORTER", 2, "CL_AREA"), _dim("PARTNER", 3, "CL_AREA"),
        _dim("PRODUCT", 4, "CL_PRODUCT"), _dim("FLOW", 5, "CL_FLOW"), _dim("RANK", 6, "CL_RANK"),
        _dim("SECTOR", 7, "CL_SECTOR"), _dim("UNIT_MEASURE", 8, "CL_UNIT"), _dim("CLASS", 9),
    ])
    # The attribute nests <Dimension> references: they must not be read as dimensions.
    attrs = ('<s:AttributeList id="AttributeDescriptor"><s:Attribute id="OBS_FLAG" assignmentStatus="Conditional">'
             '<s:ConceptIdentity><Ref id="OBS_FLAG"/></s:ConceptIdentity><s:AttributeRelationship>'
             '<s:Dimension><Ref id="FREQ"/></s:Dimension><s:Dimension><Ref id="FLOW"/></s:Dimension>'
             '</s:AttributeRelationship></s:Attribute></s:AttributeList>')
    prolog = '<!DOCTYPE m:Structure [<!ENTITY x "x">]>' if doctype else ""
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>{prolog}
<m:Structure {NS}><m:Header><m:ID>T</m:ID></m:Header><m:Structures>
<s:Dataflows><s:Dataflow id="DF_TEST" agencyID="TEST" version="1.0"><c:Name xml:lang="en">Test dataflow</c:Name>
<c:Name xml:lang="fr">Flux de test</c:Name>
<s:Structure><Ref id="DSD_TEST" agencyID="TEST" version="1.0" class="DataStructure"/></s:Structure></s:Dataflow></s:Dataflows>
<s:Codelists>{codelists}</s:Codelists>
<s:Concepts><s:ConceptScheme id="CS" agencyID="TEST" version="1.0"><c:Name xml:lang="en">Concepts</c:Name>
<s:Concept id="PARTNER"><c:Name xml:lang="en">Partner</c:Name></s:Concept>
<s:Concept id="SECTOR"><c:Name xml:lang="en">Institutional sector</c:Name></s:Concept></s:ConceptScheme></s:Concepts>
<s:DataStructures><s:DataStructure id="DSD_TEST" agencyID="TEST" version="1.0"><c:Name xml:lang="en">Test DSD</c:Name>
<s:DataStructureComponents><s:DimensionList id="DimensionDescriptor">{dims}
<s:TimeDimension id="TIME_PERIOD" position="10"><s:ConceptIdentity><Ref id="TIME_PERIOD"/></s:ConceptIdentity></s:TimeDimension>
</s:DimensionList>{attrs}<s:MeasureList id="MeasureDescriptor"><s:PrimaryMeasure id="OBS_VALUE">
<s:ConceptIdentity><Ref id="OBS_VALUE"/></s:ConceptIdentity></s:PrimaryMeasure></s:MeasureList>
</s:DataStructureComponents></s:DataStructure></s:DataStructures>
<s:Constraints><s:ContentConstraint id="CC" agencyID="TEST" version="1.0" type="Actual">
<c:Name xml:lang="en">Actual</c:Name><s:ConstraintAttachment><s:Dataflow><Ref id="DF_TEST"/></s:Dataflow></s:ConstraintAttachment>
<s:CubeRegion include="true"><c:KeyValue id="FREQ"><c:Value>A</c:Value><c:Value>M</c:Value></c:KeyValue>
<c:KeyValue id="UNIT_MEASURE"><c:Value>EUR</c:Value></c:KeyValue></s:CubeRegion></s:ContentConstraint></s:Constraints>
</m:Structures></m:Structure>'''
    return xml.encode("utf-8")


@pytest.fixture
def structure_xml() -> bytes:
    return build_structure_xml()


def recorded(name: str) -> bytes:
    """A recorded (and trimmed) real structure message, see fixtures/README.md."""
    return gzip.decompress((FIXTURES / name).read_bytes())


RECORDED = {
    "ecb_EXR.xml.gz": "EXR",
    "eurostat_une_rt_m.xml.gz": "une_rt_m",
    "lustat_DSD_INT_TRAD_DF_E4103.xml.gz": "DSD_INT_TRAD@DF_E4103",
    "ilo_DF_UNE_DEAP_SEX_AGE_RT.xml.gz": "DF_UNE_DEAP_SEX_AGE_RT",
    "bis_WS_CBPOL.xml.gz": "WS_CBPOL",
    "abs_CPI.xml.gz": "CPI",
    "imf_CPI.xml.gz": "CPI",  # codelists come from the concepts' core representations
}
