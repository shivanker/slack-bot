#! /bin/sh
set -e

pip list --not-required --format=freeze 2> /dev/null | \
  grep -vE '^(pip|setuptools|wheel)==' > requirements.txt
