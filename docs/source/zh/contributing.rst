维护与贡献
============================================================

主张对应代码与证据，中英文 RST 成对维护，命令参考从实现生成：

.. code-block:: bash

   python tools/update_cli_reference.py
   python -m sphinx -W --keep-going -b html docs/source docs/build/html
   python -m unittest discover -s tests
   python -m build
   python -m twine check dist/*

案例重建与发布见 ``MAINTENANCE.md``。文档构建不导入化学引擎、不执行计算。
公开资源不包含私有实验数据或稿件。
