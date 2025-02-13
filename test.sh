
## Dynamo
export TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
export TORCHDYNAMO_VERBOSE=1
export DYNAMO_LOG_LEVEL=DEBUG

export TG_USE_CUSTOM_OP=1

if [[ $1 == "base" ]]; then
    export EXAMPLE_TYPE=base
elif [[ $1 == "accum" ]]; then
    export EXAMPLE_TYPE=accum
elif [[ $1 == "broken" ]]; then
    export EXAMPLE_TYPE=broken
else
    echo "Invalid argument. Please provide either 'base' or 'accum' or 'broken'"
    exit 1
fi

export TG_DUMP_DIRNAME=export_dir/$EXAMPLE_TYPE

python tests/trainer/test_trainer_simple.py
