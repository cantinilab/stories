import stories
import stories.steps


def test_model_explicit():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime()


def test_model_linear():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime(quadratic=False)


def test_model_ICNN_implicit():
    step = stories.steps.ICNNImplicitStep()
    model = stories.SpaceTime()


def test_model_monge_implicit():
    step = stories.steps.MongeImplicitStep()
    model = stories.SpaceTime()
