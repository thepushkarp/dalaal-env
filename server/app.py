"""
FastAPI application for the Dalaal Browser-Use Environment.

Endpoints:
    - POST /reset: Reset the environment (pass task name in body)
    - POST /step: Execute a browser action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import DalaalEnvAction, DalaalEnvObservation
    from .dalaal_env_environment import DalaalEnvEnvironment
except (ImportError, SystemError):
    try:
        from models import DalaalEnvAction, DalaalEnvObservation
        from server.dalaal_env_environment import DalaalEnvEnvironment
    except ImportError:
        from dalaal_env.models import DalaalEnvAction, DalaalEnvObservation
        from dalaal_env.server.dalaal_env_environment import DalaalEnvEnvironment


app = create_app(
    DalaalEnvEnvironment,
    DalaalEnvAction,
    DalaalEnvObservation,
    env_name="dalaal_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
