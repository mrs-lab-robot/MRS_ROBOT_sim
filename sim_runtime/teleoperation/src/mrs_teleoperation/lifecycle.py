"""Small, simulator-independent helpers for teleoperation lifecycle reporting."""


def format_runtime_exit(
    *,
    first_app_running: bool | None,
    first_app_exiting: bool | None,
    app_running: bool,
    app_exiting: bool,
    sim_steps: int,
    received_vr: bool,
    pending_exception_type: str | None,
    pending_exception_message: str | None,
) -> str:
    """Describe why the teleoperation loop ended without conflating it with VR loss."""
    reason = "kit_shutdown_requested" if app_exiting else "kit_stopped"
    return (
        f"VR_TELEOP_EXIT reason={reason} app_running={str(app_running).lower()} "
        f"first_app_running={_format_optional_bool(first_app_running)} "
        f"first_app_exiting={_format_optional_bool(first_app_exiting)} "
        f"sim_steps={int(sim_steps)} received_vr={str(received_vr).lower()} "
        f"pending_exception={pending_exception_type or 'none'} "
        f"exception_message={pending_exception_message or 'none'}"
    )


def _format_optional_bool(value: bool | None) -> str:
    return "unknown" if value is None else str(value).lower()
