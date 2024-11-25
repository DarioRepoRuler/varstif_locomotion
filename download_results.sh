#!/bin/bash

sftp dspoljaric@eda01 << EOF
cd TAvic/outputs/graphs
lcd outputs/graphs
get -r .
quit
EOF