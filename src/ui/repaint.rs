#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RepaintDecisionInputs {
    pub(super) worker_state_changed: bool,
    pub(super) has_pending_actions: bool,
    pub(super) has_focus_or_close_command: bool,
    pub(super) has_pointer_activity: bool,
    pub(super) has_scroll_activity: bool,
    pub(super) has_input_events: bool,
    pub(super) has_active_jobs: bool,
}

pub(super) fn should_request_repaint_now(inputs: RepaintDecisionInputs) -> bool {
    inputs.worker_state_changed
        || inputs.has_pending_actions
        || inputs.has_focus_or_close_command
        || inputs.has_pointer_activity
        || inputs.has_scroll_activity
        || inputs.has_input_events
}

pub(super) fn should_request_periodic_repaint(inputs: RepaintDecisionInputs) -> bool {
    !should_request_repaint_now(inputs) && inputs.has_active_jobs
}
