import base64
from base64 import b64encode
from collections import deque
from datetime import datetime, timedelta

import pytest
import bittensor as bt
import time_machine
from bittensor.mock.wallet_mock import get_mock_hotkey, get_mock_coldkey, get_mock_keypair
from pytest_httpserver import HTTPServer

from common.protocol import PullTask, SubmitResults, Task, Feedback
from validator import Validator
from validator.config import _build_parser

FROZEN_TIME = datetime(year=2024, month=1, day=1)
TASK_COOLDOWN = 10
TASK_COOLDOWN_PENALTY = 20
COOLDOWN_VIOLATION_PENALTY = 100
COOLDOWN_VIOLATIONS_THRESHOLD = 10


@pytest.fixture()
def time_travel() -> time_machine.travel:
    traveller = time_machine.travel(FROZEN_TIME)
    traveller.start()
    yield traveller
    traveller.stop()


@pytest.fixture()
def config(make_httpserver: HTTPServer) -> bt.config:
    parser = _build_parser()
    return bt.config(
        parser,
        args=[
            "--netuid",
            "17",
            "--dataset.default_prompts_path",
            "./neurons/tests/resources/prompts.txt",
            "--generation.task_cooldown",
            f"{TASK_COOLDOWN}",
            "--generation.cooldown_penalty",
            f"{TASK_COOLDOWN_PENALTY}",
            "--neuron.cooldown_violation_penalty",
            f"{COOLDOWN_VIOLATION_PENALTY}",
            "--neuron.cooldown_violations_threshold",
            f"{COOLDOWN_VIOLATIONS_THRESHOLD}",
            "--public_api.enabled",
            f"--validation.endpoint",
            make_httpserver.url_for(""),
        ],
    )


@pytest.fixture(scope="module")
def subtensor() -> bt.MockSubtensor:
    mock = bt.MockSubtensor()
    mock.setup()
    mock.create_subnet(17)

    # Default wallet is used by the validator.
    wallet = bt.MockWallet()
    mock.force_register_neuron(
        netuid=17, hotkey=wallet.hotkey.ss58_address, coldkey=wallet.coldkey.ss58_address, stake=10000
    )

    # Miners.
    for uid in range(1, 200):
        hotkey = get_mock_hotkey(uid)
        coldkey = get_mock_coldkey(uid)
        mock.force_register_neuron(netuid=17, hotkey=hotkey, coldkey=coldkey)
    return mock


@pytest.fixture()
def validator(config: bt.config, subtensor: bt.MockSubtensor) -> Validator:
    return Validator(config, mock=True)


def create_pull_task(uid: int) -> PullTask:
    synapse = PullTask()
    synapse.dendrite.hotkey = get_mock_hotkey(uid)
    synapse.axon.hotkey = bt.MockWallet().hotkey.ss58_address
    return synapse


def create_submit_results(uid: int, task: Task) -> SubmitResults:
    keypair = get_mock_keypair(uid)
    default_wallet = bt.MockWallet()  # Validator wallet
    signature = b64encode(keypair.sign(f"{0}{task.prompt}{default_wallet.hotkey.ss58_address}{keypair.ss58_address}"))
    synapse = SubmitResults(task=task, results="dummy", submit_time=0, signature=signature)

    synapse.dendrite.hotkey = keypair.ss58_address
    synapse.axon.hotkey = default_wallet.hotkey.ss58_address
    return synapse


@pytest.mark.asyncio
async def test_pull_task(validator: Validator) -> None:
    pull = validator.pull_task(create_pull_task(1))
    assert pull.task is not None
    assert pull.task.prompt == "Monkey"

    assert validator.miners[1].assigned_task == pull.task


@pytest.mark.asyncio
async def test_pull_task_organic(validator: Validator) -> None:
    validator.task_registry.add_task("organic prompt")

    pull = validator.pull_task(create_pull_task(1))
    assert pull.task is not None
    assert pull.task.prompt == "organic prompt"

    assert validator.miners[1].assigned_task == pull.task


@pytest.mark.asyncio
async def test_pull_task_unknown_neuron(validator: Validator) -> None:
    result = validator.pull_task(create_pull_task(300))
    assert result.task is None


@pytest.mark.asyncio
async def test_pull_task_repeated_request(validator: Validator, time_travel: time_machine.travel) -> None:
    pull = validator.pull_task(create_pull_task(1))
    assert pull.task is not None

    pull = validator.pull_task(create_pull_task(1))
    assert pull.task is None
    assert pull.cooldown_until == int(time_machine.time()) + TASK_COOLDOWN

    assert validator.miners[1].assigned_task is None


@pytest.mark.asyncio
async def test_pull_task_repeated_request_if_expired(validator: Validator) -> None:
    with time_machine.travel(FROZEN_TIME):
        pull = validator.pull_task(create_pull_task(1))

    first_task = pull.task
    assert pull.task is not None
    assert validator.miners[1].assigned_task == pull.task

    with time_machine.travel(FROZEN_TIME + timedelta(hours=1)):
        pull = validator.pull_task(create_pull_task(1))

    second_task = pull.task
    assert pull.task is not None
    assert validator.miners[1].assigned_task == pull.task

    assert first_task != second_task


@pytest.mark.asyncio
async def test_pull_task_request_cooldown(validator: Validator, time_travel: time_machine.travel) -> None:
    existing_cooldown = int(time_machine.time()) + TASK_COOLDOWN
    validator.miners[1].cooldown_until = existing_cooldown
    pull = validator.pull_task(create_pull_task(1))

    assert pull.task is None
    assert validator.miners[1].cooldown_violations == 1
    assert pull.cooldown_until == existing_cooldown


async def test_pull_task_request_cooldown_penalty(validator: Validator, time_travel: time_machine.travel) -> None:
    existing_cooldown = int(time_machine.time()) + TASK_COOLDOWN
    validator.miners[1].cooldown_until = existing_cooldown
    validator.miners[1].cooldown_violations = COOLDOWN_VIOLATIONS_THRESHOLD + 1

    pull = validator.pull_task(create_pull_task(1))

    assert pull.task is None
    assert pull.cooldown_until == existing_cooldown + COOLDOWN_VIOLATION_PENALTY
    assert validator.miners[1].cooldown_until == pull.cooldown_until
    assert validator.miners[1].cooldown_violations == COOLDOWN_VIOLATIONS_THRESHOLD + 2


@pytest.mark.asyncio
async def test_submit_task(validator: Validator, httpserver: HTTPServer, time_travel: time_machine.travel) -> None:
    httpserver.expect_oneshot_request("/validate_ply/", method="POST").respond_with_json({"score": 0.71})

    pull = validator.pull_task(create_pull_task(1))
    submit = await validator.submit_results(create_submit_results(1, pull.task))

    assert submit.results == ""
    assert submit.feedback is not None
    assert submit.feedback.task_fidelity_score == 0.75
    assert submit.feedback.average_fidelity_score == 1 * 0.95 + 0.05 * 0.75
    assert submit.feedback.generations_within_8_hours == 1
    assert submit.cooldown_until == int(time_machine.time()) + TASK_COOLDOWN

    assert validator.miners[1].assigned_task is None
    assert validator.miners[1].cooldown_until == submit.cooldown_until
    assert validator.miners[1].observations == deque([int(time_machine.time())])


@pytest.mark.asyncio
async def test_submit_task_unknown_neuron(validator: Validator) -> None:
    pull = validator.pull_task(create_pull_task(1))
    submit = await validator.submit_results(create_submit_results(300, pull.task))

    assert submit.feedback is None
    assert validator.miners[1].assigned_task == pull.task  # No task reset


@pytest.mark.asyncio
async def test_submit_task_none_tasks(validator: Validator) -> None:
    pull = validator.pull_task(create_pull_task(1))
    synapse = create_submit_results(1, Task(prompt="invalid task"))
    synapse.task = None
    submit = await validator.submit_results(synapse)

    assert submit.results == ""
    assert submit.feedback == Feedback(average_fidelity_score=1)
    assert validator.miners[1].assigned_task == pull.task  # No task reset


@pytest.mark.asyncio
async def test_submit_task_wrong_task(validator: Validator) -> None:
    pull = validator.pull_task(create_pull_task(1))
    submit = await validator.submit_results(create_submit_results(1, Task(prompt="invalid task")))

    assert submit.results == ""
    assert submit.feedback == Feedback(average_fidelity_score=1)
    assert validator.miners[1].assigned_task == pull.task  # No task reset


@pytest.mark.asyncio
async def test_submit_task_skip_task(validator: Validator, time_travel: time_machine.travel) -> None:
    pull = validator.pull_task(create_pull_task(1))
    synapse = create_submit_results(1, pull.task)
    synapse.results = ""  # Task skip
    submit = await validator.submit_results(synapse)

    assert submit.feedback == Feedback(average_fidelity_score=1)
    assert validator.miners[1].assigned_task is None
    assert validator.miners[1].cooldown_until == int(time_machine.time()) + TASK_COOLDOWN


@pytest.mark.asyncio
async def test_submit_task_invalid_signature(validator: Validator) -> None:
    pull = validator.pull_task(create_pull_task(1))
    synapse = create_submit_results(1, pull.task)
    synapse.signature = base64.b64encode(get_mock_keypair(1).sign(""))
    submit = await validator.submit_results(synapse)

    assert submit.feedback == Feedback(average_fidelity_score=1)
    assert validator.miners[1].assigned_task is None
    assert validator.miners[1].cooldown_until == int(time_machine.time()) + TASK_COOLDOWN + TASK_COOLDOWN_PENALTY


@pytest.mark.asyncio
async def test_submit_task_validation_failure(validator: Validator, time_travel: time_machine.travel) -> None:
    # httpserver.expect_request("/validate_ply/", method="POST").respond_with_json({"score": 0.23})

    pull = validator.pull_task(create_pull_task(1))
    submit = await validator.submit_results(create_submit_results(1, pull.task))

    assert submit.results == ""
    assert submit.feedback is not None
    assert submit.feedback.validation_failed
    assert submit.cooldown_until == int(time_machine.time()) + TASK_COOLDOWN


@pytest.mark.asyncio
async def test_submit_task_low_fidelity(
    validator: Validator, httpserver: HTTPServer, time_travel: time_machine.travel
) -> None:
    httpserver.expect_oneshot_request("/validate_ply/", method="POST").respond_with_json({"score": 0.23})

    pull = validator.pull_task(create_pull_task(1))
    submit = await validator.submit_results(create_submit_results(1, pull.task))

    assert submit.results == ""
    assert submit.feedback is not None
    assert submit.feedback.task_fidelity_score == 0
    assert submit.cooldown_until == int(time_machine.time()) + TASK_COOLDOWN + TASK_COOLDOWN_PENALTY

    assert validator.miners[1].observations == deque([])