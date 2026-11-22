# SDMX structure fixtures

Real SDMX-ML 2.1 structure messages, recorded on 2026-10-23 and trimmed so the tests stay small and
offline. Each keeps the dataflow, its data structure definition, the concepts its dimensions use,
the codelists they reference (cut to the first 60 codes, plus the parents those codes need, so a
codelist larger than 50 codes stays larger than the enumeration limit), the HIER_CONTEXT
annotations and the content constraints. Non-English names, descriptions and other annotations
are dropped.

To record one again, save the provider's answer to the request below, then trim it:

```console
python tests/utilities/sdmx_to_buml/fixtures/minimize.py answer.xml EXR tests/utilities/sdmx_to_buml/fixtures/ecb_EXR.xml.gz
```

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

Only structural metadata is kept (codes, their names and the structure that links them), no
statistical data. It is reused under each provider's terms, with this attribution:

| Provider | Terms | Attribution |
|---|---|---|
| ECB | [ECB statistics reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html) | Source: European Central Bank, ECB Data Portal |
| Eurostat | [Eurostat copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice) (free reuse with attribution, Commission Decision 2011/833/EU) | Source: Eurostat |
| STATEC LUSTAT | [STATEC legal notice](https://statistiques.public.lu/en/support/notice.html) (CC0 1.0, with STATEC's usage guidelines) | Source: STATEC, LUSTAT |
| ILO | [ILO rights and permissions](https://www.ilo.org/rights-and-permissions) (datasets and their metadata under CC BY 4.0) | Source: International Labour Organization, ILOSTAT |
| BIS | [BIS terms and conditions](https://www.bis.org/terms_conditions.htm) | Source: Bank for International Settlements |
| ABS | [ABS copyright](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer) (CC BY 4.0) | Source: Australian Bureau of Statistics |
| IMF | [IMF copyright and usage](https://www.imf.org/en/About/copyright-and-terms) (a small excerpt kept for testing, with attribution) | Source: International Monetary Fund, © IMF |
