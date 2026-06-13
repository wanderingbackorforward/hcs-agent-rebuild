"""State manager - thin wrapper around SharedState for Agent coordination."""
from config.constants import SharedState, StateEnum


class StateManager:
    def __init__(self, shared_state: SharedState = None):
        self.state = shared_state or SharedState()

    def reset(self):
        self.state.value = StateEnum.CLASSIFY

    def set_state(self, new_state: StateEnum):
        self.state.value = new_state

    def get_state(self) -> StateEnum:
        return self.state.value

    def is_classify(self) -> bool:
        return self.state.value == StateEnum.CLASSIFY
