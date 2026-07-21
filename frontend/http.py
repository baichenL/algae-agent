# 封装前端对后端的 HTTP 请求
import requests
import os


def _headers() -> dict:
    api_key = os.getenv("ALGAE_FRONTEND_API_KEY", "")
    return {"X-API-Key": api_key} if api_key else {}

def get_last_operation(backend_url: str, timeout: int = 3):
    return requests.get(f"{backend_url}/api/v1/strain/last_operation", headers=_headers(), timeout=timeout)


def send_email(backend_url: str, payload: dict, timeout: int = 15):
    return requests.post(f"{backend_url}/api/v1/email/send", json=payload, headers=_headers(), timeout=timeout)



def submit_strain_tool(backend_url: str, tool: str, args: dict, timeout: int = 10):
    return requests.post(
        f"{backend_url}/api/v1/strain/submit_tool",
        json={"tool": tool, "args": args},
        headers=_headers(),
        timeout=timeout,
    )


def confirm_pending(backend_url: str, pending_id, approve: bool, timeout: int = 10):
    return requests.post(
        f"{backend_url}/api/v1/strain/confirm",
        json={"pending_id": pending_id, "approve": approve},
        headers=_headers(),
        timeout=timeout,
    )


def get_strain_list(backend_url: str, timeout: int = 5):
    return requests.get(f"{backend_url}/api/v1/strain/list", headers=_headers(), timeout=timeout)


def get_pending_list(backend_url: str, timeout: int = 5, status: str = "pending"):
    return requests.get(f"{backend_url}/api/v1/strain/pending", params={"status": status}, headers=_headers(), timeout=timeout)


def send_chat_message(backend_url: str, payload: dict, timeout: int = 30):
    return requests.post(f"{backend_url}/api/v1/chat", json=payload, headers=_headers(), timeout=timeout)


def start_subculture_simulation(backend_url: str, payload: dict, timeout: int = 10):
    return requests.post(
        f"{backend_url}/api/v1/workflow/simulations",
        json=payload,
        headers=_headers(),
        timeout=timeout,
    )


def get_subculture_simulation(backend_url: str, run_id: str, timeout: int = 5):
    return requests.get(
        f"{backend_url}/api/v1/workflow/simulations/{run_id}",
        headers=_headers(),
        timeout=timeout,
    )


def resolve_subculture_manual_task(
    backend_url: str,
    run_id: str,
    task_id: str,
    payload: dict,
    timeout: int = 10,
):
    return requests.post(
        f"{backend_url}/api/v1/workflow/simulations/{run_id}/manual-tasks/{task_id}/resolve",
        json=payload,
        headers=_headers(),
        timeout=timeout,
    )


def import_scientific_dataset(
    backend_url: str,
    *,
    filename: str,
    content: bytes,
    dataset_name: str,
    strain_id: str,
    mapping_json: str | None,
    timeout: int = 30,
):
    data = {"dataset_name": dataset_name, "strain_id": strain_id}
    if mapping_json:
        data["mapping_json"] = mapping_json
    return requests.post(
        f"{backend_url}/api/v1/scientific/datasets/import",
        files={"file": (filename, content)},
        data=data,
        headers=_headers(),
        timeout=timeout,
    )


def list_scientific_datasets(backend_url: str, timeout: int = 10):
    return requests.get(
        f"{backend_url}/api/v1/scientific/datasets", headers=_headers(), timeout=timeout,
    )


def start_scientific_run(backend_url: str, payload: dict, timeout: int = 90):
    return requests.post(
        f"{backend_url}/api/v1/scientific/runs", json=payload, headers=_headers(), timeout=timeout,
    )


def get_scientific_run(backend_url: str, run_id: str, timeout: int = 10):
    return requests.get(
        f"{backend_url}/api/v1/scientific/runs/{run_id}", headers=_headers(), timeout=timeout,
    )
