MCP 集成
============================================================

``surfacia-mcp`` 将本地阶段提供给 MCP 客户端，不提供云计算。程序、文件和资源
仍在你配置的环境中。

.. code-block:: json

   {"mcpServers": {"surfacia": {"command": "surfacia-mcp", "args": []}}}

使用已安装环境中的可执行文件。当前采用带 Content-Length 帧的 stdio JSON-RPC，
先核对客户端传输兼容性。实际 schema 位于 ``surfacia/mcp/tooling.py``，覆盖环境、
状态、结构、量化阶段、特征、ML 和 SHAP 启动，不包含排序。
调用会执行真实任务，应先配置工作目录与外部程序。
