from abc import abstractmethod, ABC

class AbstractAttack(ABC):
    def __init__(self, model, loss_fn, clip_min, clip_max):
        super(AbstractAttack, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    @abstractmethod
    def perturb(self, *args, **kwargs):
        """Virtual method for generating the adversarial examples.

        :param args: optional parameters used by child classes.
        :param kwargs: optional parameters used by child classes.
        :return: adversarial examples.
        """
        error = "sub-class must implement the abstractmethod:perturb"
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


