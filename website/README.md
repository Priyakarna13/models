# Unified Model Browser portal

The combined static site serves the Model Browser catalog at `/` and the
Docusaurus developer guide at `/docs`. Docusaurus consumes the repository's
top-level `docs/` directory; the assembly step adds the client-only app from
`artifact-browser/` and fingerprints its mutable assets.

Build and serve the combined portal with Node.js 20 or newer:

```bash
npm ci
npm run typecheck
npm run build:portal
npm run serve:portal
```

Open <http://localhost:4173> for the catalog and
<http://localhost:4173/docs> for documentation. A deployment workflow sets the site
origin with `DOCS_SITE_URL` and the base path with `DOCS_BASE_URL`, which is
`/models/` for the Developer Center.

Before you change `website/`, read
[artifact-browser/README.md](../artifact-browser/README.md). It holds what
`build:portal` needs, the checks to run before a push, and how the portal is deployed.
