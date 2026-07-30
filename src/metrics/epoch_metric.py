from abc import abstractmethod


class EpochMetric:
    """
    Base class for stateful epoch metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        raise NotImplementedError()
