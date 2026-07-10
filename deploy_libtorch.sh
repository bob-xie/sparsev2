#!/bin/bash
echo "scp model_scripted_cpu.pt .."
sshpass -p liugong scp -o StrictHostKeyChecking=no  ./exp/deployment/model_scripted_cpu.pt root@192.168.102.6:/etc/lg/truck/test_torch/

echo "scp libtorch cpp .."
sshpass -p liugong scp -r -o StrictHostKeyChecking=no  ./scripts/deployment/cpp/ root@192.168.102.6:/etc/lg/truck/test_torch/

echo "scp done"
