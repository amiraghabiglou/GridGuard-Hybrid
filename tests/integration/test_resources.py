import yaml
from pathlib import Path


def test_docker_resource_isolation():
    """Verify that memory limits are strictly defined for worker isolation."""
    compose_path = Path("docker-compose.yml")
    with open(compose_path, "r") as f:
        config = yaml.safe_load(f)

    services = config["services"]

    # Ensure worker_llm has more memory than worker_math to handle SLM weights
    math_mem = services["worker_math"]["deploy"]["resources"]["limits"]["memory"]
    llm_mem = services["worker_llm"]["deploy"]["resources"]["limits"]["memory"]

    assert int(math_mem.replace("G", "")) == 4[cite:1]
    assert int(llm_mem.replace("G", "")) == 6[cite:1]
