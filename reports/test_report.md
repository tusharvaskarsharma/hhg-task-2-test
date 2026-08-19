...........................F............................................ [ 98%]
.                                                                        [100%]
================================== FAILURES ===================================
__________________ test_extractive_source_ids_match_results ___________________
backend\tests\test_extractive.py:73: in test_extractive_source_ids_match_results
    RetrievalResult(doc_id="doc-abc", text="Blue whales live in oceans.", score=0.9, rank=1, source="bm25"),
    ^^^^^^^^^^^^^^^
E   NameError: name 'RetrievalResult' is not defined
=========================== short test summary info ===========================
FAILED backend/tests/test_extractive.py::test_extractive_source_ids_match_results
