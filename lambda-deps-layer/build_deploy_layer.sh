set -e

cd lambda-deps-layer || true
rm -rf build
mkdir -p build/python
cd build

pip install \
--platform manylinux2014_aarch64 \
--no-cache \
--target=python/ \
--implementation cp \
--python-version 3.11 \
--only-binary=:all: --upgrade \
-r ../requirements.txt

zip -r custom-layer.zip python

aws lambda publish-layer-version --layer-name slackbot-deps \
    --description "Python dependencies for Slack Bolt" \
    --zip-file fileb://custom-layer.zip \
    --compatible-runtimes python3.11
