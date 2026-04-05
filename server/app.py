"""FastAPI application for the DalaalEnv environment.

Endpoints:
    POST /reset — Reset the environment
    POST /step — Execute an action
    GET /state — Get current episode state
    GET /schema — Get action/observation schemas
    WS /ws — WebSocket for persistent sessions
    GET /health — Health check
    GET /metadata — Environment metadata
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    msg = "openenv is required. Install with: uv sync"
    raise ImportError(msg) from e

try:
    from dalaal_env.models import DalaalAction, DalaalObservation
    from dalaal_env.server.environment import DalaalEnvironment
except ImportError:
    from models import DalaalAction, DalaalObservation
    from server.environment import DalaalEnvironment


app = create_app(
    DalaalEnvironment,
    DalaalAction,
    DalaalObservation,
    env_name="dalaal_env",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
