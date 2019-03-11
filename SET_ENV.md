* create the virtualenv named "venv" *with python 3.6.3 64 bits*
    - virtualenv --system-site-packages -p /c/Users/kwart/AppData/Local/Programs/Python/Python36/python ./venv
    - if you don't have this version of python install it.

* activate the virtualenv
    - source ./venv/Scripts/activate

* check all packages installed in the virtualenv
    - pip list

* install all packages needed for airsim + tensorflow
    - pip install -r requirements.txt 

* deactivate the virtualenv
    - deactivate
