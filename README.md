`docker run --rm -p 8001:8000 ghcr.io/blockscout/mcp-server:latest python -m blockscout_mcp_server --http --rest --http-host 0.0.0.0`

`python agent.py`

`$body = @{ text = "What is the balance on address 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 on Ethereum chain?" } | ConvertTo-Json`
`Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8`
