
## Dynamo
export TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
export TORCHDYNAMO_VERBOSE=1
export DYNAMO_LOG_LEVEL=DEBUG

export TG_USE_CUSTOM_OP=1

# clear; pytest tests/trainer/test_trainer_simple.py::TrainerIntegrationPrerunTest::test_accumulate_loss --capture no

python tests/trainer/test_trainer_simple.py
