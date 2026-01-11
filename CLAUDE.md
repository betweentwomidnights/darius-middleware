# darius-middleware

Middleware service for the "untitled jamming app" (iOS) that handles user authentication, credit management, and proxying to Modal-hosted Magenta-RT (Darius) GPU containers.

## Project Context

This middleware sits between:
- **iOS app**: Users create loops with stable-audio-open-small, then "jam" with Magenta-RT
- **Modal backend**: GPU containers running Magenta-RT (`modal-darius`), max 3 concurrent sessions

The iOS app currently supports self-hosted HuggingFace spaces for Magenta. This middleware enables a **credits-based** system where users purchase jam time via in-app purchases and the middleware handles billing/auth.

### Why Time-Based Billing?

Unlike typical "1 credit = 1 generation" models, Magenta-RT is a **realtime jamming** system:
- User calls `/warmup` → GPU container loads model (~50s cold start)
- User calls `/jam/start` with a loop → Magenta generates continuous 4-bar chunks
- User can `/jam/stop` and `/jam/start` multiple times while container stays warm
- Container auto-scales down after 10 minutes of inactivity
- User is billed for **warm time**, not individual generations

## Architecture Overview

```
iOS App
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ darius-middleware (this service)                    │
│  - Sign in with Apple (JWT validation)              │
│  - PostgreSQL: users, credits, sessions             │
│  - Proxies requests to Modal                        │
│  - Background billing loop polls /status/peek       │
│  - Queue management for max 3 concurrent sessions   │
└─────────────────────────────────────────────────────┘
    │
    ▼
Modal (modal-darius)
    - L40S GPU containers
    - 10min scaledown window
    - modal.Dict for cheap billing peeks
```

## Phase 1 Scope (LOCAL DEVELOPMENT)

Build a working local prototype with:
- ✅ SQLite database (swap to Postgres later)
- ✅ Mock Apple auth (hardcoded test user)
- ✅ Mock Modal responses (no real GPU calls)
- ✅ Real credit deduction logic
- ✅ Background billing loop simulation

**NOT in Phase 1:**
- ❌ Real Apple JWT validation
- ❌ Real Modal API calls
- ❌ Postgres
- ❌ Docker containerization
- ❌ Queue management (assume unlimited capacity)

## Tech Stack

- **Python 3.11+**
- **FastAPI** with uvicorn
- **SQLAlchemy 2.0** (async, with aiosqlite for Phase 1)
- **Pydantic v2** for schemas
- **httpx** for async HTTP client (Modal proxy in Phase 2)
- **python-jose** for JWT handling (Phase 2+)

## Database Schema

### users
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK, auto-generated |
| apple_user_id | TEXT | Unique, from Sign in with Apple |
| email | TEXT | Optional, from Apple |
| created_at | TIMESTAMP | Default NOW() |

### credits
| Column | Type | Notes |
|--------|------|-------|
| user_id | UUID | PK, FK → users.id |
| balance_seconds | NUMERIC(12,2) | Credits as "jam seconds" |
| updated_at | TIMESTAMP | Last modification |

### jam_sessions
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| user_id | UUID | FK → users.id |
| modal_session_id | TEXT | UUID from Modal |
| billing_state | TEXT | "warming" / "active" / "closed" |
| billable_start_ts | TIMESTAMP | When warmup completed |
| billable_end_ts | TIMESTAMP | When session closed + scaledown |
| credits_charged | NUMERIC(12,2) | Running total charged |
| created_at | TIMESTAMP | |
| closed_at | TIMESTAMP | |

## API Endpoints

### Auth
```
POST /auth/mock-login
  Request: { "test_user_id": "test-user-1" }
  Response: { "token": "...", "user_id": "..." }
  Notes: Phase 1 only. Returns a simple JWT for testing.

POST /auth/apple  (Phase 2+)
  Request: { "identity_token": "..." }
  Response: { "token": "...", "user_id": "..." }
```

### Credits
```
GET /credits/balance
  Headers: Authorization: Bearer <token>
  Response: { "balance_seconds": 600.0, "formatted": "10:00" }

POST /credits/add
  Headers: Authorization: Bearer <token>
  Request: { "seconds": 600 }
  Response: { "balance_seconds": 1200.0 }
  Notes: Phase 1 test endpoint. Real IAP validation in Phase 2+.
```

### Jam Sessions (proxy to Modal)
```
POST /jam/warmup
  Headers: Authorization: Bearer <token>
  Response: {
    "ok": true,
    "session_id": "abc-123",
    "warmed": true,
    "billing_state": "active",
    "container_warmup_seconds": 49.8
  }
  Notes: Creates jam_session record, starts billing on warmup complete.

POST /jam/start
  Headers: Authorization: Bearer <token>
  Form: session_id, loop_audio (file), bpm, bars_per_chunk, ...
  Response: { "ok": true, "session_id": "...", "status": "started" }

GET /jam/next
  Headers: Authorization: Bearer <token>
  Query: session_id, timeout_seconds
  Response: { "ok": true, "chunk": { "index": 0, "audio_base64": "...", "metadata": {...} } }

POST /jam/consume
  Headers: Authorization: Bearer <token>
  Form: session_id, chunk_index
  Response: { "ok": true }

POST /jam/stop
  Headers: Authorization: Bearer <token>
  Form: session_id
  Response: { "ok": true, "status": "stopped" }

POST /jam/close
  Headers: Authorization: Bearer <token>
  Form: session_id
  Response: { "ok": true, "status": "closed", "total_charged": 185.5 }
  Notes: Finalizes billing for this session.

GET /jam/status
  Headers: Authorization: Bearer <token>
  Query: session_id
  Response: {
    "ok": true,
    "billing_state": "active",
    "billable_elapsed_seconds": 120.5,
    "credits_remaining": 479.5,
    "jam_running": true
  }
```

## Modal API Reference (for mocking)

The real Modal endpoints (see `reference/modal_darius.py`):

| Endpoint | Method | Key Response Fields |
|----------|--------|---------------------|
| /warmup | POST | session_id, warmed, container_warmup_seconds |
| /status | POST | billing_state, billable_elapsed_seconds, jam_running |
| /status/peek | GET | Same as /status but doesn't extend container lifetime |
| /jam/start | POST | ok, status |
| /jam/next | GET | chunk.index, chunk.audio_base64, chunk.metadata |
| /jam/consume | POST | ok |
| /jam/stop | POST | ok, status |
| /session/close | POST | ok, closed_ts, billing_state |

### Billing State Machine
```
warming  →  active  →  closed
   │                      │
   └──────────────────────┘
         (on error/timeout)
```

- `warming`: Container is loading model, NOT billable
- `active`: Container is warm, billing in progress
- `closed`: Session ended, billing finalized

### Key Timestamps from Modal
- `billable_start_ts`: When warmup completed (billing starts here)
- `last_activity_ts`: Last non-peek request
- `closed_last_activity_ts`: Frozen at close time
- `billable_end_estimated_ts`: last_activity + 600s scaledown window

## Mock Modal Client (Phase 1)

Create `app/modal_client.py` with a mock implementation:

```python
class MockModalClient:
    """Simulates Modal responses for local development."""
    
    def __init__(self):
        self._sessions: dict[str, dict] = {}
    
    async def warmup(self, session_id: str) -> dict:
        # Simulate 2-second warmup delay
        await asyncio.sleep(2.0)
        now = time.time()
        self._sessions[session_id] = {
            "warmed": True,
            "billing_state": "active",
            "billable_start_ts": now,
            "last_activity_ts": now,
            "jam_running": False,
        }
        return {
            "ok": True,
            "session_id": session_id,
            "warmed": True,
            "container_warmup_seconds": 2.0,
        }
    
    async def status_peek(self, session_id: str) -> dict:
        # Return current state without modifying last_activity_ts
        sess = self._sessions.get(session_id, {})
        now = time.time()
        billable_start = sess.get("billable_start_ts")
        elapsed = (now - billable_start) if billable_start else 0
        return {
            "ok": True,
            "session_id": session_id,
            "billing_state": sess.get("billing_state", "warming"),
            "billable_elapsed_seconds": elapsed,
            "jam_running": sess.get("jam_running", False),
        }
    
    # ... implement jam_start, jam_next (return fake audio), jam_stop, etc.
```

## Background Billing Loop

The middleware runs a background task that:
1. Every 5 seconds, iterates active `jam_sessions` (billing_state = "active")
2. Calls Modal `/status/peek` (mocked in Phase 1)
3. Calculates `elapsed = billable_elapsed_seconds - session.credits_charged`
4. Deducts `elapsed` from user's credit balance
5. Updates `session.credits_charged`
6. If `billing_state == "closed"`, finalizes the session

```python
async def billing_loop(db: AsyncSession, modal: ModalClient):
    while True:
        await asyncio.sleep(5.0)
        # Query active sessions
        # For each: peek status, calculate delta, deduct credits
        # Handle insufficient balance (force close session)
```

## Directory Structure

```
darius-middleware/
├── CLAUDE.md                 # This file
├── reference/
│   ├── modal_darius.py       # Real Modal implementation (read-only reference)
│   └── notes.md              # Example curl session
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, lifespan for billing loop
│   ├── config.py             # Settings via pydantic-settings
│   ├── database.py           # SQLAlchemy async engine + session
│   ├── models.py             # User, Credit, JamSession ORM models
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── auth.py               # Mock auth (Phase 1), Apple JWT (Phase 2)
│   ├── dependencies.py       # get_current_user, get_db
│   ├── modal_client.py       # MockModalClient (Phase 1), real client (Phase 2)
│   ├── billing.py            # Background billing loop
│   └── routes/
│       ├── __init__.py
│       ├── auth.py
│       ├── credits.py
│       └── jam.py
├── requirements.txt
├── .env.example
└── README.md
```

## Implementation Notes

### SQLAlchemy 2.0 Async Pattern
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    apple_user_id: Mapped[str] = mapped_column(unique=True)
    # ...
```

### FastAPI Lifespan for Background Tasks
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start billing loop
    task = asyncio.create_task(billing_loop())
    yield
    # Shutdown
    task.cancel()

app = FastAPI(lifespan=lifespan)
```

### Credit Deduction (Atomic)
```python
from sqlalchemy import update

async def deduct_credits(db: AsyncSession, user_id: UUID, amount: float) -> bool:
    result = await db.execute(
        update(Credit)
        .where(Credit.user_id == user_id)
        .where(Credit.balance_seconds >= amount)  # Prevent negative
        .values(balance_seconds=Credit.balance_seconds - amount)
    )
    await db.commit()
    return result.rowcount > 0  # False if insufficient balance
```

## Running Locally (Phase 1)

```bash
cd darius-middleware
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env from example
cp .env.example .env

# Run with auto-reload
uvicorn app.main:app --reload --port 8000

# Test endpoints
curl -X POST http://localhost:8000/auth/mock-login \
  -H "Content-Type: application/json" \
  -d '{"test_user_id": "kevin"}'

# Use returned token for subsequent requests
curl http://localhost:8000/credits/balance \
  -H "Authorization: Bearer <token>"
```

## Phase 2 Preview

Once Phase 1 is working:
1. Add `MODAL_API_KEY` to .env
2. Implement real `ModalClient` using httpx
3. Test against deployed modal-darius
4. Implement real Apple JWT validation
5. Swap SQLite → Postgres
6. Containerize with Dockerfile
7. Add to T4 docker-compose network

## iOS App Responsibilities (for Phase 2+ integration)

The iOS app is the **billing controller**. It must:

### 1. Keepalive Loop
When a session is active, the iOS app should run a background timer:
```swift
// Pseudocode
Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { _ in
    await middleware.postStatus(sessionId: currentSession, keepalive: true)
}
```
This keeps the Modal container warm AND signals to the middleware that the user is still actively using the session.

### 2. Explicit Session Close
When the user taps "End Session" (or leaves the jam UI), iOS must call `/jam/close`. This:
- Stops billing immediately
- Disables the Magenta UI
- Returns user to the "Start New Session" panel

### 3. Handle Credit Warnings
The middleware's `/jam/status` response should include `credits_remaining`. iOS should:
- Show a warning when credits drop below ~60 seconds
- Show urgent warning at ~30 seconds
- Handle forced close gracefully if credits hit zero

### 4. Reconnection After Crash
If the app crashes but the Modal container is still warm (within scaledown window):
- iOS can call `/jam/status` with the old `session_id`
- If `billing_state == "active"`, resume the keepalive loop
- If `billing_state == "closed"`, prompt user to start new session

### Future: Reduced Scaledown Window
Once the iOS keepalive is reliable, consider reducing `SCALEDOWN_WINDOW_SECONDS` in Modal from 600 (10 min) to 120-180 (2-3 min). This:
- Reduces cost for orphaned sessions
- Makes the safety net tighter
- Relies more on iOS keepalive for intentional warm-keeping

## Billing Design Decisions (Resolved)

### Q1: Should insufficient credits immediately close the Modal session?
**Answer:** Yes, with a small grace period (~10 seconds) to allow the iOS app to show a warning. The middleware should:
1. Detect balance approaching zero
2. Send a warning to the iOS app (via status response or webhook)
3. If not topped up within grace period, call `/session/close`

### Q2: What's the minimum credit balance required to start a warmup?
**Answer:** 60 seconds. This covers the ~50s warmup (which is NOT billed) plus a small buffer of actual jam time.

### Q3: Should billing continue during the scaledown window after close?
**Answer:** NO. Billing stops immediately when `/session/close` is called. The `closed_last_activity_ts` from Modal is the billing cutoff.

The Modal scaledown window (10 min, potentially reduced to 2-3 min later) is a **safety net for reconnection**, not a billing mechanism. The iOS app controls billing via:
- Keepalive loop (`/status keepalive=1` every 30-60s) → billing continues
- Explicit close (`/session/close`) → billing stops immediately

### Billing Flow Summary
```
/warmup called
    │
    ▼
billing_state = "warming" (NOT BILLED)
    │
    ▼ (model loads, ~50s)
    │
billing_state = "active", billable_start_ts set (BILLING STARTS)
    │
    ├──► iOS keepalive loop hits /status every 30-60s
    │    (extends last_activity_ts, billing continues)
    │
    ├──► User does /jam/start, /jam/stop, /jam/start...
    │    (multiple jams, still one billing session)
    │
    ▼
/session/close called
    │
    ▼
billing_state = "closed", closed_last_activity_ts frozen (BILLING STOPS)
    │
    ▼
Modal container stays warm for scaledown window (user NOT billed)
    │
    ▼
Container scales down (or user starts new session with new /warmup)
```