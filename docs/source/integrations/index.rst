MCP integration
============================================================

``surfacia-mcp`` exposes local stages to an MCP client, without hosted computation.
Programs, files and compute resources remain in your configured environment.

.. code-block:: json

   {"mcpServers": {"surfacia": {"command": "surfacia-mcp", "args": []}}}

Use the executable from the installed environment. The server currently uses
Content-Length framed stdio JSON-RPC: verify transport compatibility with your
client. Actual schemas live in ``surfacia/mcp/tooling.py``. Tools cover environment,
state, structures, quantum stages, features, ML and SHAP launch, without ranking.
Workflow calls execute real jobs; configure directories and external programs first.
