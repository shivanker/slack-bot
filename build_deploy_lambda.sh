set -e

rm -rf build
mkdir -p build/deployment_package
cd build

pip install \
--no-cache \
--target=deployment_package/ \
--implementation cp \
--python-version 3.11 \
--only-binary=:all: --upgrade \
-r ../runtime_deps.txt

cp ../lambda_function.py deployment_package/
yes | cp -r ../*.py deployment_package/

cd deployment_package
zip -r ../sushibot.zip .

cd ..
aws lambda update-function-code --function-name slackbot --zip-file fileb://sushibot.zip
