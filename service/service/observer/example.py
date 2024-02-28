from typing import List, Any
from service.observer import Observer, Observable


class Subject(Observable):
    _state: Any = None

    def set_state(self, state: Any) -> None:
        self._state = state
        self.notify_observers()

    def get_state(self) -> Any:
        return self._state


class ConcreteObserver(Observer):
    def update(self, subject: Subject) -> None:
        print(f"Observer received message: {subject.get_state()}")



if __name__ == "__main__":
    subject = Subject()
    observer1 = ConcreteObserver()
    observer2 = ConcreteObserver()

    subject.add_observer(observer1)
    subject.add_observer(observer2)

    subject.set_state("State 1")
    subject.set_state("State 2")
