# KG golden-output harness

A refactoring safety net, **not** part of the normal suite: everything here is
skipped unless `KG_GOLDEN_DIR` is set.

The KG pipeline has several order-sensitive passes (stable-sort ties, `_2`
suffixes from name reservation, first-wins attribute dedupe, warning order).
Unit tests do not notice when a refactor shifts those; byte-identical output
over a large corpus does.

    export KG_GOLDEN_DIR=/some/scratch/dir

    # 1. Collect a corpus: every KG the existing KG tests feed the pipeline.
    PYTHONPATH=. pytest -p tests.kg_golden.plugin tests/BUML/notations/kg_to_buml \
        tests/utilities tests/kg_golden/test_seed_corpus.py -q

    # 2. Record goldens at the known-good commit.
    KG_GOLDEN_MODE=record pytest tests/kg_golden/test_golden_equivalence.py -q

    # 3. After a refactor: must be byte-identical.
    pytest tests/kg_golden/test_golden_equivalence.py -q

Corpus entries are pickled `KnowledgeGraph`s, not TTL: TTL import mints random
blank-node ids that reach exporter output and consistency issue ids, so
re-importing at replay time would not be reproducible.

Re-record (step 2) only for an *intentional* behaviour change, and list the
reviewed diffs in the commit message.
