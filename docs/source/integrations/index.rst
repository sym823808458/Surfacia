Integrations
============

This section documents how Surfacia connects to external tools, automation layers, and agent-based workflows.

.. toctree::
   :maxdepth: 2

   mcp_server
   mcp_server_zh

Overview
--------

Surfacia is no longer limited to direct CLI usage. The current codebase also includes an MCP server implementation that exposes major workflow stages as agent-callable tools.

Available Integration Guides
----------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: MCP Server
      :link: mcp_server
      :link-type: doc

      Official reference for the ``surfacia-mcp`` stdio server, its tool catalog, startup flow, and debugging checklist.

   .. grid-item-card:: MCP 启动手册（中文）
      :link: mcp_server_zh
      :link-type: doc

      从零启动 Surfacia MCP server 的中文使用手册，适合第一次配置和联调时直接照着做。
