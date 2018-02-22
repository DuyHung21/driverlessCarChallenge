# Classification
The test code is in main.cpp,
The classification module is implemented in by python3.


## Complie

 1. cd to this directory
 2. g++ -Wall \`python3-config --cflags\` main.cpp -o predict \`python3-config --ldflags\`  \`pkg-config opencv --cflags --libs\`


## Run

 1. Add this directory to python path:
export PYTHONPATH=$PYTHONPATH:\`pwd\`
 2. Run
./predict test.jpg

