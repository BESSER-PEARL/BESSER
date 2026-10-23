# SDMX structure fixtures

Real SDMX-ML 2.1 structure messages, recorded on 2026-10-23 and trimmed so the tests stay small and
offline. Each keeps the dataflow, its data structure definition, the concepts its dimensions use,
the codelists they reference (cut to the first 60 codes, plus the parents those codes need, so a
codelist larger than 50 codes stays larger than the enumeration limit), the HIER_CONTEXT
annotations and the content constraints. Non-English names, descriptions and other annotations
are dropped.

| File | Provider | Request |
|---|---|---|
| `ecb_EXR.xml.gz` | ECB | `dataflow/ECB/EXR/latest?references=descendants` |
| `eurostat_une_rt_m.xml.gz` | Eurostat | `dataflow/ESTAT/une_rt_m/latest?references=descendants` |
| `lustat_DSD_INT_TRAD_DF_E4103.xml.gz` | STATEC LUSTAT | `dataflow/LU1/DSD_INT_TRAD@DF_E4103/latest?references=all` |
| `lustat_HCL_E4103.xml.gz` | STATEC LUSTAT | `hierarchicalcodelist/LU1/HCL_E4103/1.0` (untrimmed) |
| `ilo_DF_UNE_DEAP_SEX_AGE_RT.xml.gz` | ILO | `dataflow/ILO/UNE_DEAP_SEX_AGE_RT/latest?references=descendants` |
| `bis_WS_CBPOL.xml.gz` | BIS | `dataflow/BIS/WS_CBPOL/latest?references=descendants` |
| `abs_CPI.xml.gz` | ABS | `dataflow/ABS/CPI/latest?references=all` |
| `imf_CPI.xml.gz` | IMF | `dataflow/IMF.STA/CPI/latest?references=descendants` (codelists from the concepts' core representations) |

The data are published by these institutions under their own reuse terms; only structural
metadata (codes and their names) is kept here.
