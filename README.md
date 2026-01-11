# darius-middleware

Middleware service for the iOS jamming app's Magenta-RT (Darius) integration.

Handles:
- User authentication (Sign in with Apple)
- Credit management (time-based billing for jam sessions)
- Proxying to Modal-hosted GPU containers

## Quick Start (Phase 1 - Local Development)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```

## Documentation

See [CLAUDE.md](./CLAUDE.md) for comprehensive project documentation including:
- Architecture overview
- Database schema
- API endpoints
- Implementation notes
- Phase roadmap

## Reference Files

The `reference/` directory contains:
- `modal_darius.py` - The Modal backend this middleware proxies to
- `notes.md` - Example curl session showing the jam flow
