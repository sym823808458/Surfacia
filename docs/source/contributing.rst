Maintenance and contributions
============================================================

Tie claims to code and evidence. Maintain paired English/Chinese RST files.
Command reference is generated:

.. code-block:: bash

   python tools/update_cli_reference.py
   python -m sphinx -W --keep-going -b html docs/source docs/build/html
   python -m unittest discover -s tests
   python -m build
   python -m twine check dist/*

Read ``MAINTENANCE.md`` for case regeneration and publishing. Documentation
builds import no chemistry engine and execute no calculations. Private experiment
and manuscript data do not belong in public assets.
