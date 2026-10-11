#!/usr/bin/env bash
# Run every demo query against neo4j-capstone and print the output.
# Each file holds a `// graph` statement (for Neo4j Browser) and/or a
# `// table` statement; both run here, so a broken query fails loudly.
set -euo pipefail
cd "$(dirname "$0")"
CS=(docker exec -i neo4j-capstone cypher-shell -u neo4j -p phonoprobe-local --format plain)
for f in [0-9][0-9]-*.cypher; do
  echo "=================== $f"
  "${CS[@]}" < "$f"
done
