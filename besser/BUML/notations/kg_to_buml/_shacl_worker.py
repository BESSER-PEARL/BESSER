"""Child-process half of :func:`besser.BUML.notations.kg_to_buml.consistency.check_kg_consistency`.

Run as a script (``python -P _shacl_worker.py``): it reads a
pickled ``(data_graph, shapes_graph, inference)`` tuple from stdin, runs
:func:`pyshacl.validate` on it, and writes a pickled outcome to stdout:

* ``("ok", results_graph)`` -- the validation report graph;
* ``("shape_error", message)`` -- pyshacl refused the shapes graph
  (:class:`pyshacl.errors.ReportableRuntimeError`, e.g. a property shape
  without ``sh:path``).

Validation runs here, in a process of its own, because pyshacl evaluates
``sh:pattern`` with :mod:`re`, which cannot be interrupted and holds the GIL:
a catastrophically backtracking pattern written by the KG author would
otherwise pin a server thread, and every other request with it, indefinitely.
The parent kills this process when its time budget runs out.

Only :mod:`rdflib` and :mod:`pyshacl` are imported, never ``besser``, so the
child starts quickly. ``-P`` keeps this file's directory off ``sys.path``.
"""

import pickle
import sys


def _deny_network(*args, **kwargs):
    """Stand-in for ``urlopen``: a consistency check never fetches a URL."""
    raise OSError("network access is disabled during the consistency check")


def run_validation(data_graph, shapes_graph, inference):
    """Run pyshacl once and return the outcome tuple described in the module docstring.

    Also called in-process by ``consistency`` when the caller opts out of the
    child process (``timeout_seconds=None``).
    """
    import pyshacl
    from pyshacl.errors import ReportableRuntimeError

    try:
        _conforms, results_graph, _text = pyshacl.validate(
            data_graph,
            shacl_graph=shapes_graph,
            ont_graph=data_graph,
            inference=inference,
            allow_warnings=True,
            advanced=False,
            meta_shacl=False,
            debug=False,
        )
    except ReportableRuntimeError as exc:
        return ("shape_error", str(exc))
    return ("ok", results_graph)


def main() -> int:
    """Validate the graphs on stdin and write the outcome to stdout."""
    out = sys.stdout.buffer
    # Anything a library prints must not corrupt the pickle on stdout.
    sys.stdout = sys.stderr

    data_graph, shapes_graph, inference = pickle.loads(sys.stdin.buffer.read())

    import rdflib.plugins.sparql.evaluate as sparql_evaluate

    # Defence in depth behind the parent's stripping of executable SHACL:
    # SPARQL ``SERVICE`` resolves through this name with no allowlist.
    sparql_evaluate.urlopen = _deny_network

    outcome = run_validation(data_graph, shapes_graph, inference)
    out.write(pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL))
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
