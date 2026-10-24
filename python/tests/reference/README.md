Each program here runs the same computation as one of the Python tests, in C++, and writes its results to text files that `test_reference.py` compares
against the Python output. Python and C++ call the same library code, so these tests do not look for bugs in the algorithms; they catch mistakes in the
bindings themselves, such as arguments passed in the wrong order, defaults copied wrong from the headers, or arrays laid out differently than in C++. Such
mistakes usually still give a valid-looking result, so the property tests can miss them. The programs are compiled automatically by the `cpp_reference`
fixture in `conftest.py` (and cached), and the tests are skipped when no C++ compiler is found.